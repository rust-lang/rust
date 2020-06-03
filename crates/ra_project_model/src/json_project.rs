//! FIXME: write short doc here

use std::path::PathBuf;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::Deserialize;

/// Roots and crates that compose this Rust project.
#[derive(Clone, Debug, Deserialize)]
pub struct JsonProject {
    pub(crate) roots: Vec<Root>,
    pub(crate) crates: Vec<Crate>,
}

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

    // This is the preferred method of providing cfg options.
    #[serde(default)]
    pub(crate) cfg: FxHashSet<String>,

    // These two are here for transition only.
    #[serde(default)]
    pub(crate) atom_cfgs: FxHashSet<String>,
    #[serde(default)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_crate_deserialization() {
        let raw_json = json!(    {
            "crate_id": 2,
            "root_module": "this/is/a/file/path.rs",
            "deps": [
              {
                "crate": 1,
                "name": "some_dep_crate"
              },
            ],
            "edition": "2015",
            "cfg": [
              "atom_1",
              "atom_2",
              "feature=feature_1",
              "feature=feature_2",
              "other=value",
            ],

        });

        let krate: Crate = serde_json::from_value(raw_json).unwrap();

        assert!(krate.cfg.contains(&"atom_1".to_string()));
        assert!(krate.cfg.contains(&"atom_2".to_string()));
        assert!(krate.cfg.contains(&"feature=feature_1".to_string()));
        assert!(krate.cfg.contains(&"feature=feature_2".to_string()));
        assert!(krate.cfg.contains(&"other=value".to_string()));
    }

    #[test]
    fn test_crate_deserialization_old_json() {
        let raw_json = json!(    {
           "crate_id": 2,
           "root_module": "this/is/a/file/path.rs",
           "deps": [
             {
               "crate": 1,
               "name": "some_dep_crate"
             },
           ],
           "edition": "2015",
           "atom_cfgs": [
             "atom_1",
             "atom_2",
           ],
           "key_value_cfgs": {
             "feature": "feature_1",
             "feature": "feature_2",
             "other": "value",
           },
        });

        let krate: Crate = serde_json::from_value(raw_json).unwrap();

        assert!(krate.atom_cfgs.contains(&"atom_1".to_string()));
        assert!(krate.atom_cfgs.contains(&"atom_2".to_string()));
        assert!(krate.key_value_cfgs.contains_key(&"feature".to_string()));
        assert_eq!(krate.key_value_cfgs.get("feature"), Some(&"feature_2".to_string()));
        assert!(krate.key_value_cfgs.contains_key(&"other".to_string()));
        assert_eq!(krate.key_value_cfgs.get("other"), Some(&"value".to_string()));
    }
}
