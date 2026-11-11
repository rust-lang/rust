use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Serialize, Serializer};

use crate::Builder;
use crate::versions::PkgType;

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) struct Manifest {
    pub(crate) manifest_version: String,
    pub(crate) date: String,
    pub(crate) pkg: BTreeMap<String, Package>,
    pub(crate) artifacts: BTreeMap<String, Artifact>,
    pub(crate) renames: BTreeMap<String, Rename>,
    pub(crate) profiles: BTreeMap<String, Vec<String>>,
}

impl Manifest {
    pub(crate) fn add_artifact(&mut self, name: &str, f: impl FnOnce(&mut Artifact)) {
        let mut artifact = Artifact { target: BTreeMap::new() };
        f(&mut artifact);
        self.artifacts.insert(name.to_string(), artifact);
    }
}

#[derive(Serialize)]
pub(crate) struct Package {
    pub(crate) version: String,
    pub(crate) git_commit_hash: Option<String>,
    pub(crate) target: BTreeMap<String, Target>,
}

#[derive(Serialize)]
pub(crate) struct Rename {
    pub(crate) to: String,
}

#[derive(Serialize)]
pub(crate) struct Artifact {
    pub(crate) target: BTreeMap<String, Vec<ArtifactFile>>,
}

impl Artifact {
    pub(crate) fn add_file(&mut self, builder: &mut Builder, target: &str, path: &str) {
        if let Some(path) = record_shipped_file(builder, builder.input.join(path)) {
            self.target.entry(target.into()).or_insert_with(Vec::new).push(ArtifactFile {
                url: builder.url(&path),
                hash_sha256: FileHash::Missing(path),
            });
        }
    }

    pub(crate) fn add_tarball(&mut self, builder: &mut Builder, target: &str, base_path: &str) {
        let files = self.target.entry(target.into()).or_insert_with(Vec::new);
        let base_path = builder.input.join(base_path);
        for compression in &["gz", "xz"] {
            if let Some(tarball) = tarball_variant(builder, &base_path, compression) {
                files.push(ArtifactFile {
                    url: builder.url(&tarball),
                    hash_sha256: FileHash::Missing(tarball),
                });
            }
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) struct ArtifactFile {
    pub(crate) url: String,
    pub(crate) hash_sha256: FileHash,
}

#[derive(Serialize, Default)]
pub(crate) struct Target {
    pub(crate) available: bool,
    pub(crate) url: Option<String>,
    pub(crate) hash: Option<FileHash>,
    pub(crate) xz_url: Option<String>,
    pub(crate) xz_hash: Option<FileHash>,
    pub(crate) components: Option<Vec<Component>>,
    pub(crate) extensions: Option<Vec<Component>>,
}

impl Target {
    pub(crate) fn from_compressed_tar(builder: &mut Builder, base_path: &str) -> Self {
        let base_path = builder.input.join(base_path);
        let gz = tarball_variant(builder, &base_path, "gz");
        let xz = tarball_variant(builder, &base_path, "xz");

        if gz.is_none() && xz.is_none() {
            return Self::unavailable();
        }

        Self {
            available: true,
            components: None,
            extensions: None,
            // .gz
            url: gz.as_ref().map(|path| builder.url(path)),
            hash: gz.map(FileHash::Missing),
            // .xz
            xz_url: xz.as_ref().map(|path| builder.url(path)),
            xz_hash: xz.map(FileHash::Missing),
        }
    }

    pub(crate) fn unavailable() -> Self {
        Self::default()
    }
}

#[derive(Serialize)]
pub(crate) struct Component {
    pub(crate) pkg: String,
    pub(crate) target: String,
}

impl Component {
    pub(crate) fn from_pkg(pkg: &PkgType, target: &str) -> Self {
        Self { pkg: pkg.manifest_component_name(), target: target.to_string() }
    }
}

#[allow(unused)]
pub(crate) enum FileHash {
    Missing(PathBuf),
    Present(String),
}

impl Serialize for FileHash {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            FileHash::Missing(path) => Err(serde::ser::Error::custom(format!(
                "can't serialize a missing hash for file {}",
                path.display()
            ))),
            FileHash::Present(inner) => inner.serialize(serializer),
        }
    }
}

fn tarball_variant(builder: &mut Builder, base: &Path, ext: &str) -> Option<PathBuf> {
    let mut path = base.to_path_buf();
    path.set_extension(ext);
    record_shipped_file(builder, path)
}

fn record_shipped_file(builder: &mut Builder, path: PathBuf) -> Option<PathBuf> {
    if path.is_file() {
        builder.shipped_files.insert(
            path.file_name()
                .expect("missing filename")
                .to_str()
                .expect("non-utf-8 filename")
                .to_string(),
        );
        Some(path)
    } else {
        None
    }
}

pub(crate) fn visit_file_hashes(manifest: &mut Manifest, mut f: impl FnMut(&mut FileHash)) {
    for pkg in manifest.pkg.values_mut() {
        for target in pkg.target.values_mut() {
            if let Some(hash) = &mut target.hash {
                f(hash);
            }
            if let Some(hash) = &mut target.xz_hash {
                f(hash);
            }
        }
    }

    for artifact in manifest.artifacts.values_mut() {
        for target in artifact.target.values_mut() {
            for file in target {
                f(&mut file.hash_sha256);
            }
        }
    }
}
