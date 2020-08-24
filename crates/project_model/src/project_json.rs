//! FIXME: write short doc here

use std::path::PathBuf;

use base_db::{CrateId, CrateName, Dependency, Edition};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashMap;
use serde::{de, Deserialize};

use crate::cfg_flag::CfgFlag;

/// Roots and crates that compose this Rust project.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectJson {
    pub(crate) crates: Vec<Crate>,
}

/// A crate points to the root module of a crate and lists the dependencies of the crate. This is
/// useful in creating the crate graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Crate {
    pub(crate) root_module: AbsPathBuf,
    pub(crate) edition: Edition,
    pub(crate) deps: Vec<Dependency>,
    pub(crate) cfg: Vec<CfgFlag>,
    pub(crate) target: Option<String>,
    pub(crate) env: FxHashMap<String, String>,
    pub(crate) proc_macro_dylib_path: Option<AbsPathBuf>,
    pub(crate) is_workspace_member: bool,
    pub(crate) include: Vec<AbsPathBuf>,
    pub(crate) exclude: Vec<AbsPathBuf>,
}

impl ProjectJson {
    pub fn new(base: &AbsPath, data: ProjectJsonData) -> ProjectJson {
        ProjectJson {
            crates: data
                .crates
                .into_iter()
                .map(|crate_data| {
                    let is_workspace_member = crate_data.is_workspace_member.unwrap_or_else(|| {
                        crate_data.root_module.is_relative()
                            && !crate_data.root_module.starts_with("..")
                            || crate_data.root_module.starts_with(base)
                    });
                    let root_module = base.join(crate_data.root_module);
                    let (include, exclude) = match crate_data.source {
                        Some(src) => {
                            let absolutize = |dirs: Vec<PathBuf>| {
                                dirs.into_iter().map(|it| base.join(it)).collect::<Vec<_>>()
                            };
                            (absolutize(src.include_dirs), absolutize(src.exclude_dirs))
                        }
                        None => (vec![root_module.parent().unwrap().to_path_buf()], Vec::new()),
                    };

                    Crate {
                        root_module,
                        edition: crate_data.edition.into(),
                        deps: crate_data
                            .deps
                            .into_iter()
                            .map(|dep_data| Dependency {
                                crate_id: CrateId(dep_data.krate as u32),
                                name: dep_data.name,
                            })
                            .collect::<Vec<_>>(),
                        cfg: crate_data.cfg,
                        target: crate_data.target,
                        env: crate_data.env,
                        proc_macro_dylib_path: crate_data
                            .proc_macro_dylib_path
                            .map(|it| base.join(it)),
                        is_workspace_member,
                        include,
                        exclude,
                    }
                })
                .collect::<Vec<_>>(),
        }
    }
}

#[derive(Deserialize)]
pub struct ProjectJsonData {
    crates: Vec<CrateData>,
}

#[derive(Deserialize)]
struct CrateData {
    root_module: PathBuf,
    edition: EditionData,
    deps: Vec<DepData>,
    #[serde(default)]
    cfg: Vec<CfgFlag>,
    target: Option<String>,
    #[serde(default)]
    env: FxHashMap<String, String>,
    proc_macro_dylib_path: Option<PathBuf>,
    is_workspace_member: Option<bool>,
    source: Option<CrateSource>,
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

#[derive(Deserialize)]
struct CrateSource {
    include_dirs: Vec<PathBuf>,
    exclude_dirs: Vec<PathBuf>,
}

fn deserialize_crate_name<'de, D>(de: D) -> Result<CrateName, D::Error>
where
    D: de::Deserializer<'de>,
{
    let name = String::deserialize(de)?;
    CrateName::new(&name).map_err(|err| de::Error::custom(format!("invalid crate name: {:?}", err)))
}
