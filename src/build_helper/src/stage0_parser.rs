use std::collections::BTreeMap;

#[derive(Default, Clone)]
pub struct Stage0 {
    pub compiler: VersionMetadata,
    pub rustfmt: Option<VersionMetadata>,
    pub config: Stage0Config,
    pub checksums_sha256: BTreeMap<String, String>,
}

#[derive(Default, Clone)]
pub struct VersionMetadata {
    pub date: String,
    pub version: String,
}

#[derive(Default, Clone)]
pub struct Stage0Config {
    pub dist_server: String,
    pub artifacts_server: String,
    pub artifacts_with_llvm_assertions_server: String,
    pub git_merge_commit_email: String,
    pub nightly_branch: String,
}

pub fn parse_stage0_file() -> Stage0 {
    let stage0_content = include_str!("../../stage0");

    let mut stage0 = Stage0::default();
    for line in stage0_content.lines() {
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        // Ignore comments
        if line.starts_with('#') {
            continue;
        }

        let (key, value) = line.split_once('=').unwrap();

        match key {
            "dist_server" => stage0.config.dist_server = value.to_owned(),
            "artifacts_server" => stage0.config.artifacts_server = value.to_owned(),
            "artifacts_with_llvm_assertions_server" => {
                stage0.config.artifacts_with_llvm_assertions_server = value.to_owned()
            }
            "git_merge_commit_email" => stage0.config.git_merge_commit_email = value.to_owned(),
            "nightly_branch" => stage0.config.nightly_branch = value.to_owned(),

            "compiler_date" => stage0.compiler.date = value.to_owned(),
            "compiler_version" => stage0.compiler.version = value.to_owned(),

            "rustfmt_date" => {
                stage0.rustfmt.get_or_insert(VersionMetadata::default()).date = value.to_owned();
            }
            "rustfmt_version" => {
                stage0.rustfmt.get_or_insert(VersionMetadata::default()).version = value.to_owned();
            }

            dist if dist.starts_with("dist") => {
                stage0.checksums_sha256.insert(key.to_owned(), value.to_owned());
            }

            unsupported => {
                println!("'{unsupported}' field is not supported.");
            }
        }
    }

    stage0
}
