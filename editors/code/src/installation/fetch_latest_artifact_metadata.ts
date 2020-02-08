import fetch from "node-fetch";
import { GithubRepo, ArtifactMetadata } from "./interfaces";

const GITHUB_API_ENDPOINT_URL = "https://api.github.com";

export interface FetchLatestArtifactMetadataOpts {
    repo: GithubRepo;
    artifactFileName: string;
}

export async function fetchLatestArtifactMetadata(
    opts: FetchLatestArtifactMetadataOpts
): Promise<null | ArtifactMetadata> {

    const repoOwner = encodeURIComponent(opts.repo.owner);
    const repoName  = encodeURIComponent(opts.repo.name);

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
        downloadUrl: artifact.browser_download_url
    };

    // Noise denotes tremendous amount of data that we are not using here
    interface GithubRelease {
        name: string;
        assets: Array<{
            browser_download_url: string;

            [noise: string]: unknown;
        }>;

        [noise: string]: unknown;
    }

}
