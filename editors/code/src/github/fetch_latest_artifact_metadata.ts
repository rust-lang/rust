import fetch from "node-fetch";

const GITHUB_API_ENDPOINT_URL = "https://api.github.com";

export interface FetchLatestArtifactMetadataOpts {
    repoName: string;
    repoOwner: string;
    artifactFileName: string;
}

export interface ArtifactMetadata {
    releaseName: string;
    releaseDate: Date;
    downloadUrl: string;
}

export async function fetchLatestArtifactMetadata(
    opts: FetchLatestArtifactMetadataOpts
): Promise<ArtifactMetadata | null> {

    const repoOwner = encodeURIComponent(opts.repoOwner);
    const repoName  = encodeURIComponent(opts.repoName);

    const apiEndpointPath = `/repos/${repoOwner}/${repoName}/releases/latest`;
    const requestUrl = GITHUB_API_ENDPOINT_URL + apiEndpointPath;

    // We skip runtime type checks for simplicity (here we cast from `any` to `Release`)

    const response: GithubRelease = await fetch(requestUrl, {
            headers: { Accept: "application/vnd.github.v3+json" }
        })
        .then(res => res.json());

    const artifact = response.assets.find(artifact => artifact.name === opts.artifactFileName);

    return !artifact ? null : {
        releaseName: response.name,
        releaseDate: new Date(response.published_at),
        downloadUrl: artifact.browser_download_url
    };

    // Noise denotes tremendous amount of data that we are not using here
    interface GithubRelease {
        name: string;
        published_at: Date;
        assets: Array<{
            browser_download_url: string;

            [noise: string]: unknown;
        }>;

        [noise: string]: unknown;
    }

}
