use serde::Serialize;
use std::collections::BTreeMap;

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
