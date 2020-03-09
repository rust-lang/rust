export interface GithubRepo {
    name: string;
    owner: string;
}

/**
 * Metadata about particular artifact retrieved from GitHub releases.
 */
export interface ArtifactReleaseInfo {
    releaseDate: Date;
    releaseName: string;
    downloadUrl: string;
}

/**
 * Represents the source of a an artifact which is either specified by the user
 * explicitly, or bundled by this extension from GitHub releases.
 */
export type ArtifactSource = ArtifactSource.ExplicitPath | ArtifactSource.GithubRelease;

export namespace ArtifactSource {
    /**
     * Type tag for `ArtifactSource` discriminated union.
     */
    export const enum Type { ExplicitPath, GithubRelease }

    export interface ExplicitPath {
        type: Type.ExplicitPath;

        /**
         * Filesystem path to the binary specified by the user explicitly.
         */
        path: string;
    }

    export interface GithubRelease {
        type: Type.GithubRelease;

        /**
         * Repository where the binary is stored.
         */
        repo: GithubRepo;


        // FIXME: add installationPath: string;

        /**
         * Directory on the filesystem where the bundled binary is stored.
         */
        dir: string;

        /**
         * Name of the binary file. It is stored under the same name on GitHub releases
         * and in local `.dir`.
         */
        file: string;

        /**
         * Tag of github release that denotes a version required by this extension.
         */
        tag: string;
    }
}
