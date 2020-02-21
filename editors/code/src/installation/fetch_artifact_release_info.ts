import fetch from "node-fetch";
import { GithubRepo, ArtifactReleaseInfo } from "./interfaces";
import { log } from "../util";

const GITHUB_API_ENDPOINT_URL = "https://api.github.com";


/**
 * Fetches the release with `releaseTag` (or just latest release when not specified)
 * from GitHub `repo` and returns metadata about `artifactFileName` shipped with
 * this release or `null` if no such artifact was published.
 */
export async function fetchArtifactReleaseInfo(
    repo: GithubRepo, artifactFileName: string, releaseTag?: string
): Promise<null | ArtifactReleaseInfo> {

    const repoOwner = encodeURIComponent(repo.owner);
    const repoName = encodeURIComponent(repo.name);

    const apiEndpointPath = releaseTag
        ? `/repos/${repoOwner}/${repoName}/releases/tags/${releaseTag}`
        : `/repos/${repoOwner}/${repoName}/releases/latest`;

    const requestUrl = GITHUB_API_ENDPOINT_URL + apiEndpointPath;

    // We skip runtime type checks for simplicity (here we cast from `any` to `GithubRelease`)

    log.debug("Issuing request for released artifacts metadata to", requestUrl);

    // FIXME: handle non-ok response
    const response: GithubRelease = await fetch(requestUrl, {
        headers: { Accept: "application/vnd.github.v3+json" }
    })
        .then(res => res.json());

    const artifact = response.assets.find(artifact => artifact.name === artifactFileName);

    if (!artifact) return null;

    return {
        releaseName: response.name,
        downloadUrl: artifact.browser_download_url
    };

    // We omit declaration of tremendous amount of fields that we are not using here
    interface GithubRelease {
        name: string;
        assets: Array<{
            name: string;
            // eslint-disable-next-line camelcase
            browser_download_url: string;
        }>;
    }
}
