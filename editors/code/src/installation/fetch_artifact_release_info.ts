import fetch from "node-fetch";
import { GithubRepo, ArtifactReleaseInfo } from "./interfaces";
import { log } from "../util";

const GITHUB_API_ENDPOINT_URL = "https://api.github.com";

/**
 * Fetches the release with `releaseTag` from GitHub `repo` and
 * returns metadata about `artifactFileName` shipped with
 * this release.
 *
 * @throws Error upon network failure or if no such repository, release, or artifact exists.
 */
export async function fetchArtifactReleaseInfo(
    repo: GithubRepo,
    artifactFileName: string,
    releaseTag: string
): Promise<ArtifactReleaseInfo> {

    const repoOwner = encodeURIComponent(repo.owner);
    const repoName = encodeURIComponent(repo.name);

    const apiEndpointPath = `/repos/${repoOwner}/${repoName}/releases/tags/${releaseTag}`;

    const requestUrl = GITHUB_API_ENDPOINT_URL + apiEndpointPath;

    log.debug("Issuing request for released artifacts metadata to", requestUrl);

    const response = await fetch(requestUrl, { headers: { Accept: "application/vnd.github.v3+json" } });

    if (!response.ok) {
        log.error("Error fetching artifact release info", {
            requestUrl,
            releaseTag,
            artifactFileName,
            response: {
                headers: response.headers,
                status: response.status,
                body: await response.text(),
            }
        });

        throw new Error(
            `Got response ${response.status} when trying to fetch ` +
            `"${artifactFileName}" artifact release info for ${releaseTag} release`
        );
    }

    // We skip runtime type checks for simplicity (here we cast from `any` to `GithubRelease`)
    const release: GithubRelease = await response.json();

    const artifact = release.assets.find(artifact => artifact.name === artifactFileName);

    if (!artifact) throw new Error(
        `Artifact ${artifactFileName} was not found in ${release.name} release!`
    );

    return {
        releaseName: release.name,
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
