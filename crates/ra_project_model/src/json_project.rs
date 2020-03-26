//! FIXME: write short doc here

use std::path::PathBuf;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::Deserialize;

/// A root points to the directory which contains Rust crates. rust-analyzer watches all files in
/// all roots. Roots might be nested.
#[derive(Clone, Debug, Deserialize)]
#[serde(transparent)]
pub struct Root {
    pub(crate) path: PathBuf,
}

/// A crate points to the root module of a crate and lists the dependencies of the crate. This is
/// useful in creating the crate graph.
#[derive(Clone, Debug, Deserialize)]
pub struct Crate {
    pub(crate) root_module: PathBuf,
    pub(crate) edition: Edition,
    pub(crate) deps: Vec<Dep>,
    pub(crate) atom_cfgs: FxHashSet<String>,
    pub(crate) key_value_cfgs: FxHashMap<String, String>,
    pub(crate) out_dir: Option<PathBuf>,
    pub(crate) proc_macro_dylib_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename = "edition")]
pub enum Edition {
    #[serde(rename = "2015")]
    Edition2015,
    #[serde(rename = "2018")]
    Edition2018,
}

/// Identifies a crate by position in the crates array.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[serde(transparent)]
pub struct CrateId(pub usize);

/// A dependency of a crate, identified by its id in the crates array and name.
#[derive(Clone, Debug, Deserialize)]
pub struct Dep {
    #[serde(rename = "crate")]
    pub(crate) krate: CrateId,
    pub(crate) name: String,
}

/// Roots and crates that compose this Rust project.
#[derive(Clone, Debug, Deserialize)]
pub struct JsonProject {
    pub(crate) roots: Vec<Root>,
    pub(crate) crates: Vec<Crate>,
}
