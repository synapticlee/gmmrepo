%[labels,GMModel,errFlag] = clusterGMM(NClust,mName,expDate,expNumStr,taskName,NPlanes,startType, doPlots, whichTasks)

%written 2019-08-23
%clusters TM vs SW activity statistics using a GMM

%GMModel initialises using kmeans++
%vs. normal k-means, k-means++ picks better initial centres,
%by iteratively choosing those with sufficient distance from the chosen ones
% otherwise startType = 'known', use prior knowledge that two clusters should be along the axes

%pi is the proportion each cluster is found in the population ("prior")
%mu and cov are the data mu and covariance of the neurons assigned to each cluster
%gamma is the "responsibilities" i.e. posterior, normalized to integrate(sum) to 1

%example:
%    clusterGMM(3,'JL035','2019-06-15',...
%    '1045_1047_1048_1049',...
%    {'TM', 'blankball', 'SW', 'blankwheel'}, ...
%    2, 'default', 1, {'TM', 'SW'});

function [labels,GMModel,errFlag] = clusterGMM(NClust,mName,expDate,expNumStr,taskName,NPlanes,startType, doPlots, whichTasks)

    if nargin < 9
        whichTasks = {'TM', 'SW'};
        if nargin < 8
            doPlots = true;
            if nargin < 7
                startType = 'default';  %alt: known
                if nargin < 6
                    NPlanes = 2;
                    if nargin == 0
                        NClust = 3;
                    end
                end
            end
        end
    end

    errFlag     = 0;
    doPlotsIso  = false;
    NIters      = 10; %iterations for GMM fitting

    clustFN = fullfile(loadDirs, 'isolationDists', ...
                sprintf('clusterlabels_N%d_%s_%s_%s.mat', ...
                NClust,mName,expDate,expNumStr));

    %get isolation distance per condition
    eachDB  = calcIsolationDists(mName,expDate,expNumStr,taskName, NPlanes, doPlotsIso); 

    %subselect for chosen tasks
    allDB   = horzcat(eachDB{strcmp(taskName, whichTasks{1})}, ...
                      eachDB{strcmp(taskName, whichTasks{2})}); 

    if exist(clustFN)
        load(clustFN)
    else %run GMM 
        switch startType 
        case 'default'
            %fitgmdist picks the iter with the lowest negative log likelihood
            GMModel = fitgmdist(allDB, NClust, 'Replicates', 10, 'Options', statset('MaxIter', 500)); 
        case 'known' %% initialize using hard-coded guesses for centers,covs based on data
            try
                for itr = 1:NIters
                    doStartPlots    = false;
                    S{itr}          = generateStart(allDB,NClust,doStartPlots); %this is not used as default initialization works quite well
                    GMModel         = fitgmdist(allDB, NClust, 'Start', S{itr}); 
                    nllIter(itr)    = GMModel.NegativeLogLikelihood;
                end
                bestItr = find(nllIter==min(nllIter));
                GMModel = fitgmdist(allDB, NClust, 'Start', S{bestItr(1)}); %(1) in case multiple matches
            catch 
                error('generateStart.m may not be included - try the default initialization first')
            end
        otherwise
            warning('define choice for initialization')
        end

        %predict labels
        pi = GMModel.ComponentProportion; %mixing proportions

        for cl = 1:NClust
            mu          = GMModel.mu(cl,:);
            cov         = GMModel.Sigma(:,:,cl);
            likelihood  = mvnpdf(allDB,mu,cov);
            preds(:,cl) = pi(cl)*likelihood;
        end

        [~,labels] = max(preds,[],2); %choose cluster label according to probabiltiies 

    end

    if doPlots
        axisLabels  = whichTasks;
        taskColors  = {[0.9, 0.3 0.1],[0 0.6 0.6]}; %orange, blue
        cmap        = [taskColors{1}; 0.8*ones(1,3); taskColors{2}; 0.5*ones(1,3)];

        figure('name', 'gmm fits', 'Position', [600 500 330 320])
        scatter(allDB(:,1),allDB(:,2), 30, 0.8*ones(1,3),'filled', 'MarkerEdgeColor', 'k')
        axis square
        axis([0 4.5 0 4.5])
        hold on;
        for cl = 1:NClust
            mu  = GMModel.mu(cl,:);
            cov = GMModel.Sigma(:,:,cl);
            gm  = gmdistribution(mu,cov);
            x = allDB(:,1); 
            y = allDB(:,2); 
            gmPDF = @(x,y)reshape(pdf(gm,[x(:) y(:)]),size(x));
            g = gca;
            fcontour(gmPDF,[g.XLim g.YLim])
        end
        xlabel(sprintf('isolation distance (%s)',axisLabels{1})); 
        ylabel(sprintf('isolation distance (%s)',axisLabels{2})); 
        colormap(cmocean('amp'))
        caxis([0 1])
        set(gca, 'FontSize', 12)


        figure('name', 'cluster asssignments', 'Position', [600 500 330 320])
        scatter(allDB(:,1),allDB(:,2), 30, labels, 'filled', 'MarkerEdgeColor', 'k')
        axis square
        xlabel(sprintf('isolation distance (%s)',axisLabels{1})); 
        ylabel(sprintf('isolation distance (%s)',axisLabels{2})); 
        xticks(0:4); yticks(0:4)
        axis([0 4.5 0 4.5])
        colormap(gca,cmap)
        caxis(gca, [1 4])
        set(gca, 'FontSize', 12)
    end

    if ~exist(clustFN) && NClust < 4 %don't save N=4 unless relabelling is robust
        save(clustFN, 'labels', 'GMModel')
    end

end
