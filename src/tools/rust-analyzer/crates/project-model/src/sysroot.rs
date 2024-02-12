//! Loads "sysroot" crate.
//!
//! One confusing point here is that normally sysroot is a bunch of `.rlib`s,
//! but we can't process `.rlib` and need source code instead. The source code
//! is typically installed with `rustup component add rust-src` command.

use std::{env, fs, iter, ops, path::PathBuf, process::Command};

use anyhow::{format_err, Context, Result};
use base_db::CrateName;
use itertools::Itertools;
use la_arena::{Arena, Idx};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashMap;

use crate::{utf8_stdout, CargoConfig, CargoWorkspace, ManifestPath};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Sysroot {
    root: AbsPathBuf,
    src_root: AbsPathBuf,
    mode: SysrootMode,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum SysrootMode {
    Workspace(CargoWorkspace),
    Stitched(Stitched),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct Stitched {
    crates: Arena<SysrootCrateData>,
}

impl ops::Index<SysrootCrate> for Stitched {
    type Output = SysrootCrateData;
    fn index(&self, index: SysrootCrate) -> &SysrootCrateData {
        &self.crates[index]
    }
}

impl Stitched {
    pub(crate) fn public_deps(&self) -> impl Iterator<Item = (CrateName, SysrootCrate, bool)> + '_ {
        // core is added as a dependency before std in order to
        // mimic rustcs dependency order
        ["core", "alloc", "std"]
            .into_iter()
            .zip(iter::repeat(true))
            .chain(iter::once(("test", false)))
            .filter_map(move |(name, prelude)| {
                Some((CrateName::new(name).unwrap(), self.by_name(name)?, prelude))
            })
    }

    pub(crate) fn proc_macro(&self) -> Option<SysrootCrate> {
        self.by_name("proc_macro")
    }

    pub(crate) fn crates(&self) -> impl ExactSizeIterator<Item = SysrootCrate> + '_ {
        self.crates.iter().map(|(id, _data)| id)
    }

    fn by_name(&self, name: &str) -> Option<SysrootCrate> {
        let (id, _data) = self.crates.iter().find(|(_id, data)| data.name == name)?;
        Some(id)
    }
}

pub(crate) type SysrootCrate = Idx<SysrootCrateData>;

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct SysrootCrateData {
    pub(crate) name: String,
    pub(crate) root: ManifestPath,
    pub(crate) deps: Vec<SysrootCrate>,
}

impl Sysroot {
    /// Returns sysroot "root" directory, where `bin/`, `etc/`, `lib/`, `libexec/`
    /// subfolder live, like:
    /// `$HOME/.rustup/toolchains/nightly-2022-07-23-x86_64-unknown-linux-gnu`
    pub fn root(&self) -> &AbsPath {
        &self.root
    }

    /// Returns the sysroot "source" directory, where stdlib sources are located, like:
    /// `$HOME/.rustup/toolchains/nightly-2022-07-23-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library`
    pub fn src_root(&self) -> &AbsPath {
        &self.src_root
    }

    pub fn is_empty(&self) -> bool {
        match &self.mode {
            SysrootMode::Workspace(ws) => ws.packages().next().is_none(),
            SysrootMode::Stitched(stitched) => stitched.crates.is_empty(),
        }
    }

    pub fn loading_warning(&self) -> Option<String> {
        let has_core = match &self.mode {
            SysrootMode::Workspace(ws) => ws.packages().any(|p| ws[p].name == "core"),
            SysrootMode::Stitched(stitched) => stitched.by_name("core").is_some(),
        };
        if !has_core {
            let var_note = if env::var_os("RUST_SRC_PATH").is_some() {
                " (`RUST_SRC_PATH` might be incorrect, try unsetting it)"
            } else {
                " try running `rustup component add rust-src` to possible fix this"
            };
            Some(format!(
                "could not find libcore in loaded sysroot at `{}`{var_note}",
                self.src_root.as_path(),
            ))
        } else {
            None
        }
    }

    pub fn num_packages(&self) -> usize {
        match &self.mode {
            SysrootMode::Workspace(ws) => ws.packages().count(),
            SysrootMode::Stitched(c) => c.crates().count(),
        }
    }

    pub(crate) fn mode(&self) -> &SysrootMode {
        &self.mode
    }
}

// FIXME: Expose a builder api as loading the sysroot got way too modular and complicated.
impl Sysroot {
    /// Attempts to discover the toolchain's sysroot from the given `dir`.
    pub fn discover(
        dir: &AbsPath,
        extra_env: &FxHashMap<String, String>,
        metadata: bool,
    ) -> Result<Sysroot> {
        tracing::debug!("discovering sysroot for {dir}");
        let sysroot_dir = discover_sysroot_dir(dir, extra_env)?;
        let sysroot_src_dir =
            discover_sysroot_src_dir_or_add_component(&sysroot_dir, dir, extra_env)?;
        Ok(Sysroot::load(sysroot_dir, sysroot_src_dir, metadata))
    }

    pub fn discover_with_src_override(
        current_dir: &AbsPath,
        extra_env: &FxHashMap<String, String>,
        src: AbsPathBuf,
        metadata: bool,
    ) -> Result<Sysroot> {
        tracing::debug!("discovering sysroot for {current_dir}");
        let sysroot_dir = discover_sysroot_dir(current_dir, extra_env)?;
        Ok(Sysroot::load(sysroot_dir, src, metadata))
    }

    pub fn discover_rustc_src(&self) -> Option<ManifestPath> {
        get_rustc_src(&self.root)
    }

    pub fn discover_rustc(&self) -> anyhow::Result<AbsPathBuf> {
        let rustc = self.root.join("bin/rustc");
        tracing::debug!(?rustc, "checking for rustc binary at location");
        match fs::metadata(&rustc) {
            Ok(_) => Ok(rustc),
            Err(e) => Err(e).context(format!(
                "failed to discover rustc in sysroot: {:?}",
                AsRef::<std::path::Path>::as_ref(&self.root)
            )),
        }
    }

    pub fn with_sysroot_dir(sysroot_dir: AbsPathBuf, metadata: bool) -> Result<Sysroot> {
        let sysroot_src_dir = discover_sysroot_src_dir(&sysroot_dir).ok_or_else(|| {
            format_err!("can't load standard library from sysroot path {sysroot_dir}")
        })?;
        Ok(Sysroot::load(sysroot_dir, sysroot_src_dir, metadata))
    }

    pub fn load(sysroot_dir: AbsPathBuf, sysroot_src_dir: AbsPathBuf, metadata: bool) -> Sysroot {
        if metadata {
            let sysroot: Option<_> = (|| {
                let sysroot_cargo_toml = ManifestPath::try_from(
                    AbsPathBuf::try_from(&*format!("{sysroot_src_dir}/sysroot/Cargo.toml")).ok()?,
                )
                .ok()?;
                let current_dir =
                    AbsPathBuf::try_from(&*format!("{sysroot_src_dir}/sysroot")).ok()?;
                let res = CargoWorkspace::fetch_metadata(
                    &sysroot_cargo_toml,
                    &current_dir,
                    &CargoConfig::default(),
                    &|_| (),
                )
                .map_err(|e| {
                    tracing::error!(
                        "failed to load sysroot `{sysroot_src_dir}/sysroot/Cargo.toml`: {}",
                        e
                    );
                    e
                });
                if let Err(e) =
                    std::fs::remove_file(format!("{sysroot_src_dir}/sysroot/Cargo.lock"))
                {
                    tracing::error!(
                        "failed to remove sysroot `{sysroot_src_dir}/sysroot/Cargo.lock`: {}",
                        e
                    )
                }
                let mut res = res.ok()?;

                // Patch out `rustc-std-workspace-*` crates to point to the real crates.
                // This is done prior to `CrateGraph` construction to avoid having duplicate `std` targets.

                let mut fake_core = None;
                let mut fake_alloc = None;
                let mut fake_std = None;
                let mut real_core = None;
                let mut real_alloc = None;
                let mut real_std = None;
                res.packages.iter().enumerate().for_each(|(idx, package)| {
                    match package.name.strip_prefix("rustc-std-workspace-") {
                        Some("core") => fake_core = Some((idx, package.id.clone())),
                        Some("alloc") => fake_alloc = Some((idx, package.id.clone())),
                        Some("std") => fake_std = Some((idx, package.id.clone())),
                        Some(_) => {
                            tracing::warn!("unknown rustc-std-workspace-* crate: {}", package.name)
                        }
                        None => match &*package.name {
                            "core" => real_core = Some(package.id.clone()),
                            "alloc" => real_alloc = Some(package.id.clone()),
                            "std" => real_std = Some(package.id.clone()),
                            _ => (),
                        },
                    }
                });

                let patches =
                    [fake_core.zip(real_core), fake_alloc.zip(real_alloc), fake_std.zip(real_std)]
                        .into_iter()
                        .flatten();

                let resolve = res.resolve.as_mut().expect("metadata executed with deps");
                let mut remove_nodes = vec![];
                for (idx, node) in resolve.nodes.iter_mut().enumerate() {
                    // Replace them in the dependency list
                    node.deps.iter_mut().for_each(|dep| {
                        if let Some((_, real)) =
                            patches.clone().find(|((_, fake_id), _)| *fake_id == dep.pkg)
                        {
                            dep.pkg = real;
                        }
                    });
                    if patches.clone().any(|((_, fake), _)| fake == node.id) {
                        remove_nodes.push(idx);
                    }
                }
                // Remove the fake ones from the resolve data
                remove_nodes.into_iter().rev().for_each(|r| {
                    resolve.nodes.remove(r);
                });
                // Remove the fake ones from the packages
                patches.map(|((r, _), _)| r).sorted().rev().for_each(|r| {
                    res.packages.remove(r);
                });

                res.workspace_members = res
                    .packages
                    .iter()
                    .filter(|&package| RELEVANT_SYSROOT_CRATES.contains(&&*package.name))
                    .map(|package| package.id.clone())
                    .collect();
                let cargo_workspace = CargoWorkspace::new(res);
                Some(Sysroot {
                    root: sysroot_dir.clone(),
                    src_root: sysroot_src_dir.clone(),
                    mode: SysrootMode::Workspace(cargo_workspace),
                })
            })();
            if let Some(sysroot) = sysroot {
                return sysroot;
            }
        }
        let mut stitched = Stitched { crates: Arena::default() };

        for path in SYSROOT_CRATES.trim().lines() {
            let name = path.split('/').last().unwrap();
            let root = [format!("{path}/src/lib.rs"), format!("lib{path}/lib.rs")]
                .into_iter()
                .map(|it| sysroot_src_dir.join(it))
                .filter_map(|it| ManifestPath::try_from(it).ok())
                .find(|it| fs::metadata(it).is_ok());

            if let Some(root) = root {
                stitched.crates.alloc(SysrootCrateData {
                    name: name.into(),
                    root,
                    deps: Vec::new(),
                });
            }
        }

        if let Some(std) = stitched.by_name("std") {
            for dep in STD_DEPS.trim().lines() {
                if let Some(dep) = stitched.by_name(dep) {
                    stitched.crates[std].deps.push(dep)
                }
            }
        }

        if let Some(alloc) = stitched.by_name("alloc") {
            for dep in ALLOC_DEPS.trim().lines() {
                if let Some(dep) = stitched.by_name(dep) {
                    stitched.crates[alloc].deps.push(dep)
                }
            }
        }

        if let Some(proc_macro) = stitched.by_name("proc_macro") {
            for dep in PROC_MACRO_DEPS.trim().lines() {
                if let Some(dep) = stitched.by_name(dep) {
                    stitched.crates[proc_macro].deps.push(dep)
                }
            }
        }
        Sysroot {
            root: sysroot_dir,
            src_root: sysroot_src_dir,
            mode: SysrootMode::Stitched(stitched),
        }
    }
}

fn discover_sysroot_dir(
    current_dir: &AbsPath,
    extra_env: &FxHashMap<String, String>,
) -> Result<AbsPathBuf> {
    let mut rustc = Command::new(toolchain::rustc());
    rustc.envs(extra_env);
    rustc.current_dir(current_dir).args(["--print", "sysroot"]);
    tracing::debug!("Discovering sysroot by {:?}", rustc);
    let stdout = utf8_stdout(rustc)?;
    Ok(AbsPathBuf::assert(PathBuf::from(stdout)))
}

fn discover_sysroot_src_dir(sysroot_path: &AbsPathBuf) -> Option<AbsPathBuf> {
    if let Ok(path) = env::var("RUST_SRC_PATH") {
        if let Ok(path) = AbsPathBuf::try_from(path.as_str()) {
            let core = path.join("core");
            if fs::metadata(&core).is_ok() {
                tracing::debug!("Discovered sysroot by RUST_SRC_PATH: {path}");
                return Some(path);
            }
            tracing::debug!("RUST_SRC_PATH is set, but is invalid (no core: {core:?}), ignoring");
        } else {
            tracing::debug!("RUST_SRC_PATH is set, but is invalid, ignoring");
        }
    }

    get_rust_src(sysroot_path)
}

fn discover_sysroot_src_dir_or_add_component(
    sysroot_path: &AbsPathBuf,
    current_dir: &AbsPath,
    extra_env: &FxHashMap<String, String>,
) -> Result<AbsPathBuf> {
    discover_sysroot_src_dir(sysroot_path)
        .or_else(|| {
            let mut rustup = Command::new(toolchain::rustup());
            rustup.envs(extra_env);
            rustup.current_dir(current_dir).args(["component", "add", "rust-src"]);
            tracing::info!("adding rust-src component by {:?}", rustup);
            utf8_stdout(rustup).ok()?;
            get_rust_src(sysroot_path)
        })
        .ok_or_else(|| {
            format_err!(
                "\
can't load standard library from sysroot
{sysroot_path}
(discovered via `rustc --print sysroot`)
try installing the Rust source the same way you installed rustc",
            )
        })
}

fn get_rustc_src(sysroot_path: &AbsPath) -> Option<ManifestPath> {
    let rustc_src = sysroot_path.join("lib/rustlib/rustc-src/rust/compiler/rustc/Cargo.toml");
    let rustc_src = ManifestPath::try_from(rustc_src).ok()?;
    tracing::debug!("checking for rustc source code: {rustc_src}");
    if fs::metadata(&rustc_src).is_ok() {
        Some(rustc_src)
    } else {
        None
    }
}

fn get_rust_src(sysroot_path: &AbsPath) -> Option<AbsPathBuf> {
    let rust_src = sysroot_path.join("lib/rustlib/src/rust/library");
    tracing::debug!("checking sysroot library: {rust_src}");
    if fs::metadata(&rust_src).is_ok() {
        Some(rust_src)
    } else {
        None
    }
}

const SYSROOT_CRATES: &str = "
alloc
backtrace
core
panic_abort
panic_unwind
proc_macro
profiler_builtins
std
stdarch/crates/std_detect
test
unwind";

const ALLOC_DEPS: &str = "core";

const STD_DEPS: &str = "
alloc
panic_unwind
panic_abort
core
profiler_builtins
unwind
std_detect
test";

// core is required for our builtin derives to work in the proc_macro lib currently
const PROC_MACRO_DEPS: &str = "
std
core";

const RELEVANT_SYSROOT_CRATES: &[&str] = &["core", "alloc", "std", "test", "proc_macro"];
