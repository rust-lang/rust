use crate::Builder;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) struct Manifest {
    pub(crate) manifest_version: String,
    pub(crate) date: String,
    pub(crate) pkg: BTreeMap<String, Package>,
    pub(crate) renames: BTreeMap<String, Rename>,
    pub(crate) profiles: BTreeMap<String, Vec<String>>,
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

#[derive(Serialize, Default)]
pub(crate) struct Target {
    pub(crate) available: bool,
    pub(crate) url: Option<String>,
    pub(crate) hash: Option<String>,
    pub(crate) xz_url: Option<String>,
    pub(crate) xz_hash: Option<String>,
    pub(crate) components: Option<Vec<Component>>,
    pub(crate) extensions: Option<Vec<Component>>,
}

impl Target {
    pub(crate) fn from_compressed_tar(builder: &Builder, base_path: &str) -> Self {
        let base_path = builder.input.join(base_path);
        let gz = Self::tarball_variant(&base_path, "gz");
        let xz = Self::tarball_variant(&base_path, "xz");

        if gz.is_none() {
            return Self::unavailable();
        }

        Self {
            available: true,
            components: None,
            extensions: None,
            // .gz
            url: gz.as_ref().map(|path| builder.url(path)),
            hash: gz.map(|path| Self::digest_of(builder, &path)),
            // .xz
            xz_url: xz.as_ref().map(|path| builder.url(path)),
            xz_hash: xz.map(|path| Self::digest_of(builder, &path)),
        }
    }

    fn tarball_variant(base: &Path, ext: &str) -> Option<PathBuf> {
        let mut path = base.to_path_buf();
        path.set_extension(ext);
        if path.is_file() { Some(path) } else { None }
    }

    fn digest_of(builder: &Builder, path: &Path) -> String {
        // TEMPORARY CODE -- DON'T REVIEW :)
        let file_name = path.file_name().unwrap().to_str().unwrap();
        builder.digests.get(file_name).unwrap().clone()
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
    pub(crate) fn from_str(pkg: &str, target: &str) -> Self {
        Self { pkg: pkg.to_string(), target: target.to_string() }
    }
}
