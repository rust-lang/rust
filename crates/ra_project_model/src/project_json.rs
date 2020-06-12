//! FIXME: write short doc here

use std::path::PathBuf;

use paths::{AbsPath, AbsPathBuf};
use ra_cfg::CfgOptions;
use ra_db::{CrateId, CrateName, Dependency, Edition};
use rustc_hash::FxHashSet;
use serde::{de, Deserialize};
use stdx::split_delim;

/// Roots and crates that compose this Rust project.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectJson {
    pub(crate) roots: Vec<Root>,
    pub(crate) crates: Vec<Crate>,
}

/// A root points to the directory which contains Rust crates. rust-analyzer watches all files in
/// all roots. Roots might be nested.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Root {
    pub(crate) path: AbsPathBuf,
}

/// A crate points to the root module of a crate and lists the dependencies of the crate. This is
/// useful in creating the crate graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Crate {
    pub(crate) root_module: AbsPathBuf,
    pub(crate) edition: Edition,
    pub(crate) deps: Vec<Dependency>,
    pub(crate) cfg: CfgOptions,
    pub(crate) target: Option<String>,
    pub(crate) out_dir: Option<AbsPathBuf>,
    pub(crate) proc_macro_dylib_path: Option<AbsPathBuf>,
}

impl ProjectJson {
    pub fn new(base: &AbsPath, data: ProjectJsonData) -> ProjectJson {
        ProjectJson {
            roots: data.roots.into_iter().map(|path| Root { path: base.join(path) }).collect(),
            crates: data
                .crates
                .into_iter()
                .map(|crate_data| Crate {
                    root_module: base.join(crate_data.root_module),
                    edition: crate_data.edition.into(),
                    deps: crate_data
                        .deps
                        .into_iter()
                        .map(|dep_data| Dependency {
                            crate_id: CrateId(dep_data.krate as u32),
                            name: dep_data.name,
                        })
                        .collect::<Vec<_>>(),
                    cfg: {
                        let mut cfg = CfgOptions::default();
                        for entry in &crate_data.cfg {
                            match split_delim(entry, '=') {
                                Some((key, value)) => {
                                    cfg.insert_key_value(key.into(), value.into());
                                }
                                None => cfg.insert_atom(entry.into()),
                            }
                        }
                        cfg
                    },
                    target: crate_data.target,
                    out_dir: crate_data.out_dir.map(|it| base.join(it)),
                    proc_macro_dylib_path: crate_data.proc_macro_dylib_path.map(|it| base.join(it)),
                })
                .collect::<Vec<_>>(),
        }
    }
}

#[derive(Deserialize)]
pub struct ProjectJsonData {
    roots: Vec<PathBuf>,
    crates: Vec<CrateData>,
}

#[derive(Deserialize)]
struct CrateData {
    root_module: PathBuf,
    edition: EditionData,
    deps: Vec<DepData>,
    #[serde(default)]
    cfg: FxHashSet<String>,
    target: Option<String>,
    out_dir: Option<PathBuf>,
    proc_macro_dylib_path: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(rename = "edition")]
enum EditionData {
    #[serde(rename = "2015")]
    Edition2015,
    #[serde(rename = "2018")]
    Edition2018,
}

impl From<EditionData> for Edition {
    fn from(data: EditionData) -> Self {
        match data {
            EditionData::Edition2015 => Edition::Edition2015,
            EditionData::Edition2018 => Edition::Edition2018,
        }
    }
}

#[derive(Deserialize)]
struct DepData {
    /// Identifies a crate by position in the crates array.
    #[serde(rename = "crate")]
    krate: usize,
    #[serde(deserialize_with = "deserialize_crate_name")]
    name: CrateName,
}

fn deserialize_crate_name<'de, D>(de: D) -> Result<CrateName, D::Error>
where
    D: de::Deserializer<'de>,
{
    let name = String::deserialize(de)?;
    CrateName::new(&name).map_err(|err| de::Error::custom(format!("invalid crate name: {:?}", err)))
}
