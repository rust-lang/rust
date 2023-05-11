//! `rust-project.json` file format.
//!
//! This format is spiritually a serialization of [`base_db::CrateGraph`]. The
//! idea here is that people who do not use Cargo, can instead teach their build
//! system to generate `rust-project.json` which can be ingested by
//! rust-analyzer.
//!
//! This short file is a somewhat big conceptual piece of the architecture of
//! rust-analyzer, so it's worth elaborating on the underlying ideas and
//! motivation.
//!
//! For rust-analyzer to function, it needs some information about the project.
//! Specifically, it maintains an in-memory data structure which lists all the
//! crates (compilation units) and dependencies between them. This is necessary
//! a global singleton, as we do want, eg, find usages to always search across
//! the whole project, rather than just in the "current" crate.
//!
//! Normally, we get this "crate graph" by calling `cargo metadata
//! --message-format=json` for each cargo workspace and merging results. This
//! works for your typical cargo project, but breaks down for large folks who
//! have a monorepo with an infinite amount of Rust code which is built with bazel or
//! some such.
//!
//! To support this use case, we need to make _something_ configurable. To avoid
//! a [midlayer mistake](https://lwn.net/Articles/336262/), we allow configuring
//! the lowest possible layer. `ProjectJson` is essentially a hook to just set
//! that global singleton in-memory data structure. It is optimized for power,
//! not for convenience (you'd be using cargo anyway if you wanted nice things,
//! right? :)
//!
//! `rust-project.json` also isn't necessary a file. Architecturally, we support
//! any convenient way to specify this data, which today is:
//!
//! * file on disk
//! * a field in the config (ie, you can send a JSON request with the contents
//!   of rust-project.json to rust-analyzer, no need to write anything to disk)
//!
//! Another possible thing we don't do today, but which would be totally valid,
//! is to add an extension point to VS Code extension to register custom
//! project.
//!
//! In general, it is assumed that if you are going to use `rust-project.json`,
//! you'd write a fair bit of custom code gluing your build system to ra through
//! this JSON format. This logic can take form of a VS Code extension, or a
//! proxy process which injects data into "configure" LSP request, or maybe just
//! a simple build system rule to generate the file.
//!
//! In particular, the logic for lazily loading parts of the monorepo as the
//! user explores them belongs to that extension (it's totally valid to change
//! rust-project.json over time via configuration request!)

use std::path::PathBuf;

use base_db::{CrateDisplayName, CrateId, CrateName, Dependency, Edition};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashMap;
use serde::{de, Deserialize};

use crate::cfg_flag::CfgFlag;

/// Roots and crates that compose this Rust project.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectJson {
    /// e.g. `path/to/sysroot`
    pub(crate) sysroot: Option<AbsPathBuf>,
    /// e.g. `path/to/sysroot/lib/rustlib/src/rust`
    pub(crate) sysroot_src: Option<AbsPathBuf>,
    project_root: AbsPathBuf,
    crates: Vec<Crate>,
}

/// A crate points to the root module of a crate and lists the dependencies of the crate. This is
/// useful in creating the crate graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Crate {
    pub(crate) display_name: Option<CrateDisplayName>,
    pub(crate) root_module: AbsPathBuf,
    pub(crate) edition: Edition,
    pub(crate) version: Option<String>,
    pub(crate) deps: Vec<Dependency>,
    pub(crate) cfg: Vec<CfgFlag>,
    pub(crate) target: Option<String>,
    pub(crate) env: FxHashMap<String, String>,
    pub(crate) proc_macro_dylib_path: Option<AbsPathBuf>,
    pub(crate) is_workspace_member: bool,
    pub(crate) include: Vec<AbsPathBuf>,
    pub(crate) exclude: Vec<AbsPathBuf>,
    pub(crate) is_proc_macro: bool,
    pub(crate) repository: Option<String>,
}

impl ProjectJson {
    /// Create a new ProjectJson instance.
    ///
    /// # Arguments
    ///
    /// * `base` - The path to the workspace root (i.e. the folder containing `rust-project.json`)
    /// * `data` - The parsed contents of `rust-project.json`, or project json that's passed via
    ///            configuration.
    pub fn new(base: &AbsPath, data: ProjectJsonData) -> ProjectJson {
        ProjectJson {
            sysroot: data.sysroot.map(|it| base.join(it)),
            sysroot_src: data.sysroot_src.map(|it| base.join(it)),
            project_root: base.to_path_buf(),
            crates: data
                .crates
                .into_iter()
                .map(|crate_data| {
                    let is_workspace_member = crate_data.is_workspace_member.unwrap_or_else(|| {
                        crate_data.root_module.is_relative()
                            && !crate_data.root_module.starts_with("..")
                            || crate_data.root_module.starts_with(base)
                    });
                    let root_module = base.join(crate_data.root_module).normalize();
                    let (include, exclude) = match crate_data.source {
                        Some(src) => {
                            let absolutize = |dirs: Vec<PathBuf>| {
                                dirs.into_iter()
                                    .map(|it| base.join(it).normalize())
                                    .collect::<Vec<_>>()
                            };
                            (absolutize(src.include_dirs), absolutize(src.exclude_dirs))
                        }
                        None => (vec![root_module.parent().unwrap().to_path_buf()], Vec::new()),
                    };

                    Crate {
                        display_name: crate_data
                            .display_name
                            .map(CrateDisplayName::from_canonical_name),
                        root_module,
                        edition: crate_data.edition.into(),
                        version: crate_data.version.as_ref().map(ToString::to_string),
                        deps: crate_data
                            .deps
                            .into_iter()
                            .map(|dep_data| {
                                Dependency::new(dep_data.name, CrateId(dep_data.krate as u32))
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
                        is_proc_macro: crate_data.is_proc_macro,
                        repository: crate_data.repository,
                    }
                })
                .collect::<Vec<_>>(),
        }
    }

    /// Returns the number of crates in the project.
    pub fn n_crates(&self) -> usize {
        self.crates.len()
    }

    /// Returns an iterator over the crates in the project.
    pub fn crates(&self) -> impl Iterator<Item = (CrateId, &Crate)> + '_ {
        self.crates.iter().enumerate().map(|(idx, krate)| (CrateId(idx as u32), krate))
    }

    /// Returns the path to the project's root folder.
    pub fn path(&self) -> &AbsPath {
        &self.project_root
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct ProjectJsonData {
    sysroot: Option<PathBuf>,
    sysroot_src: Option<PathBuf>,
    crates: Vec<CrateData>,
}

#[derive(Deserialize, Debug, Clone)]
struct CrateData {
    display_name: Option<String>,
    root_module: PathBuf,
    edition: EditionData,
    #[serde(default)]
    version: Option<semver::Version>,
    deps: Vec<DepData>,
    #[serde(default)]
    cfg: Vec<CfgFlag>,
    target: Option<String>,
    #[serde(default)]
    env: FxHashMap<String, String>,
    proc_macro_dylib_path: Option<PathBuf>,
    is_workspace_member: Option<bool>,
    source: Option<CrateSource>,
    #[serde(default)]
    is_proc_macro: bool,
    #[serde(default)]
    repository: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename = "edition")]
enum EditionData {
    #[serde(rename = "2015")]
    Edition2015,
    #[serde(rename = "2018")]
    Edition2018,
    #[serde(rename = "2021")]
    Edition2021,
}

impl From<EditionData> for Edition {
    fn from(data: EditionData) -> Self {
        match data {
            EditionData::Edition2015 => Edition::Edition2015,
            EditionData::Edition2018 => Edition::Edition2018,
            EditionData::Edition2021 => Edition::Edition2021,
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
struct DepData {
    /// Identifies a crate by position in the crates array.
    #[serde(rename = "crate")]
    krate: usize,
    #[serde(deserialize_with = "deserialize_crate_name")]
    name: CrateName,
}

#[derive(Deserialize, Debug, Clone)]
struct CrateSource {
    include_dirs: Vec<PathBuf>,
    exclude_dirs: Vec<PathBuf>,
}

fn deserialize_crate_name<'de, D>(de: D) -> Result<CrateName, D::Error>
where
    D: de::Deserializer<'de>,
{
    let name = String::deserialize(de)?;
    CrateName::new(&name).map_err(|err| de::Error::custom(format!("invalid crate name: {err:?}")))
}
