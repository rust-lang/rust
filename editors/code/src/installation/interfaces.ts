export interface GithubRepo {
    name: string;
    owner: string;
}

/**
 * Metadata about particular artifact retrieved from GitHub releases.
 */
export interface ArtifactMetadata {
    releaseName: string;
    downloadUrl: string;
}

/**
 * Type tag for `BinarySource` discriminated union.
 */
export enum BinarySourceType { ExplicitPath, GithubBinary }

/**
 * Represents the source of a binary artifact which is either specified by the user
 * explicitly, or bundled by this extension from GitHub releases.
 */
export type BinarySource = ExplicitPathSource | GithubBinarySource;


export interface ExplicitPathSource {
    type: BinarySourceType.ExplicitPath;

    /**
     * Filesystem path to the binary specified by the user explicitly.
     */
    path: string;
}

export interface GithubBinarySource {
    type: BinarySourceType.GithubBinary;

    /**
     * Repository where the binary is stored.
     */
    repo: GithubRepo;

    /**
     * Directory on the filesystem where the bundled binary is stored.
     */
    dir: string;

    /**
     * Name of the binary file. It is stored under the same name on GitHub releases
     * and in local `.dir`.
     */
    file: string;
}
