export interface GithubRepo {
    name: string;
    owner: string;
}

export interface ArtifactMetadata {
    releaseName: string;
    downloadUrl: string;
}


export enum BinarySourceType { ExplicitPath, GithubBinary }

export type BinarySource = EplicitPathSource | GithubBinarySource;

export interface EplicitPathSource {
    type: BinarySourceType.ExplicitPath;
    path: string;
}

export interface GithubBinarySource {
    type: BinarySourceType.GithubBinary;
    repo: GithubRepo;
    dir: string;
    file: string;
}
