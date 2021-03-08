% these are hard-coded assumptions for cluster initializations (means and covariances) based on exploratory data analysis
% default initializations based on kmeans++ (default in fitgmdist) works quite well, so only use this if the default fit fails badly
function S = generateStart(allDB,NClust, doPlots)
    if nargin < 3
        doPlots = true
    end
    NDim = size(allDB,2);
    S.mu = 1 + rand(NClust,NDim); %NClust x NTasks
    S.mu(1,1) = (rand(1)+0.1)*0.5;
    S.mu(1,2) = 1 + (4-1) * rand(1);

    S.mu(2,1) = 1 + (4-1) * rand(1);
    S.mu(2,2) = (rand(1)+0.1)*0.5;

    % diags
    S.Sigma(1,1,1) = (1-0.1) * rand(1) + 0.1;    
    S.Sigma(2,2,1) = (2-1) * rand(1) + 1;

    % off diags
    S.Sigma(1,2,1) = (S.Sigma(1,1,1)-1e-5-0.05) * rand(1) + 0.05; %some fraction
    S.Sigma(2,1,1) = S.Sigma(1,2,1);

    % diags
    S.Sigma(2,2,2) = (1-0.1) * rand(1) + 0.1;
    S.Sigma(1,1,2) = (2-1) * rand(1) + 1;

    % off diags
    S.Sigma(1,2,2) = (S.Sigma(2,2,2)-1e-5-0.05) * rand(1) + 0.05;
    S.Sigma(2,1,2) = S.Sigma(1,2,2);


    for cl = NDim+1:NClust
        S.Sigma(:,:,cl) = eye(NDim,NDim).*(0.3+rand(NDim,NDim)) + 0.05*rand(1);
    end

    if NClust > 3
        S.mu(NClust,:) = rand(1,2)*0.5;
    end

    S.ComponentProportion = ones(1,NClust)*1/NClust; %uniform

    if doPlots
        figure('Name', 'initialized clusters', 'Position', [120 500 900 420])
        axis square
        axis([0 4 0 4])
        hold on;
        for cl = 1:NClust
            mu  = S.mu(cl,:);
            cov = S.Sigma(:,:,cl);
            gm  = gmdistribution(mu,cov);
            x = 0:0.2:4;
            y = 0:0.2:4;
            gmPDF = @(x,y)reshape(pdf(gm,[x(:) y(:)]),size(x));
            g = gca;
            fcontour(gmPDF,[g.XLim g.YLim])
        end
    end
end
