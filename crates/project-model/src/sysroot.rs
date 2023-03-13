//! Loads "sysroot" crate.
//!
//! One confusing point here is that normally sysroot is a bunch of `.rlib`s,
//! but we can't process `.rlib` and need source code instead. The source code
//! is typically installed with `rustup component add rust-src` command.

use std::{env, fs, iter, ops, path::PathBuf, process::Command};

use anyhow::{format_err, Result};
use base_db::CrateName;
use la_arena::{Arena, Idx};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashMap;

use crate::{utf8_stdout, ManifestPath};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Sysroot {
    root: AbsPathBuf,
    src_root: AbsPathBuf,
    crates: Arena<SysrootCrateData>,
}

pub(crate) type SysrootCrate = Idx<SysrootCrateData>;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SysrootCrateData {
    pub name: String,
    pub root: ManifestPath,
    pub deps: Vec<SysrootCrate>,
}

impl ops::Index<SysrootCrate> for Sysroot {
    type Output = SysrootCrateData;
    fn index(&self, index: SysrootCrate) -> &SysrootCrateData {
        &self.crates[index]
    }
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

    pub fn public_deps(&self) -> impl Iterator<Item = (CrateName, SysrootCrate, bool)> + '_ {
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

    pub fn proc_macro(&self) -> Option<SysrootCrate> {
        self.by_name("proc_macro")
    }

    pub fn crates(&self) -> impl Iterator<Item = SysrootCrate> + ExactSizeIterator + '_ {
        self.crates.iter().map(|(id, _data)| id)
    }

    pub fn is_empty(&self) -> bool {
        self.crates.is_empty()
    }
}

// FIXME: Expose a builder api as loading the sysroot got way too modular and complicated.
impl Sysroot {
    /// Attempts to discover the toolchain's sysroot from the given `dir`.
    pub fn discover(dir: &AbsPath, extra_env: &FxHashMap<String, String>) -> Result<Sysroot> {
        tracing::debug!("discovering sysroot for {}", dir.display());
        let sysroot_dir = discover_sysroot_dir(dir, extra_env)?;
        let sysroot_src_dir =
            discover_sysroot_src_dir_or_add_component(&sysroot_dir, dir, extra_env)?;
        Ok(Sysroot::load(sysroot_dir, sysroot_src_dir))
    }

    pub fn discover_with_src_override(
        current_dir: &AbsPath,
        extra_env: &FxHashMap<String, String>,
        src: AbsPathBuf,
    ) -> Result<Sysroot> {
        tracing::debug!("discovering sysroot for {}", current_dir.display());
        let sysroot_dir = discover_sysroot_dir(current_dir, extra_env)?;
        Ok(Sysroot::load(sysroot_dir, src))
    }

    pub fn discover_rustc(&self) -> Option<ManifestPath> {
        get_rustc_src(&self.root)
    }

    pub fn with_sysroot_dir(sysroot_dir: AbsPathBuf) -> Result<Sysroot> {
        let sysroot_src_dir = discover_sysroot_src_dir(&sysroot_dir).ok_or_else(|| {
            format_err!("can't load standard library from sysroot {}", sysroot_dir.display())
        })?;
        Ok(Sysroot::load(sysroot_dir, sysroot_src_dir))
    }

    pub fn load(sysroot_dir: AbsPathBuf, sysroot_src_dir: AbsPathBuf) -> Sysroot {
        let mut sysroot =
            Sysroot { root: sysroot_dir, src_root: sysroot_src_dir, crates: Arena::default() };

        for path in SYSROOT_CRATES.trim().lines() {
            let name = path.split('/').last().unwrap();
            let root = [format!("{path}/src/lib.rs"), format!("lib{path}/lib.rs")]
                .into_iter()
                .map(|it| sysroot.src_root.join(it))
                .filter_map(|it| ManifestPath::try_from(it).ok())
                .find(|it| fs::metadata(it).is_ok());

            if let Some(root) = root {
                sysroot.crates.alloc(SysrootCrateData {
                    name: name.into(),
                    root,
                    deps: Vec::new(),
                });
            }
        }

        if let Some(std) = sysroot.by_name("std") {
            for dep in STD_DEPS.trim().lines() {
                if let Some(dep) = sysroot.by_name(dep) {
                    sysroot.crates[std].deps.push(dep)
                }
            }
        }

        if let Some(alloc) = sysroot.by_name("alloc") {
            for dep in ALLOC_DEPS.trim().lines() {
                if let Some(dep) = sysroot.by_name(dep) {
                    sysroot.crates[alloc].deps.push(dep)
                }
            }
        }

        if let Some(proc_macro) = sysroot.by_name("proc_macro") {
            for dep in PROC_MACRO_DEPS.trim().lines() {
                if let Some(dep) = sysroot.by_name(dep) {
                    sysroot.crates[proc_macro].deps.push(dep)
                }
            }
        }

        if sysroot.by_name("core").is_none() {
            let var_note = if env::var_os("RUST_SRC_PATH").is_some() {
                " (`RUST_SRC_PATH` might be incorrect, try unsetting it)"
            } else {
                ""
            };
            tracing::error!(
                "could not find libcore in sysroot path `{}`{}",
                sysroot.src_root.as_path().display(),
                var_note,
            );
        }

        sysroot
    }

    fn by_name(&self, name: &str) -> Option<SysrootCrate> {
        let (id, _data) = self.crates.iter().find(|(_id, data)| data.name == name)?;
        Some(id)
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
                tracing::debug!("Discovered sysroot by RUST_SRC_PATH: {}", path.display());
                return Some(path);
            }
            tracing::debug!("RUST_SRC_PATH is set, but is invalid (no core: {:?}), ignoring", core);
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
{}
(discovered via `rustc --print sysroot`)
try installing the Rust source the same way you installed rustc",
                sysroot_path.display(),
            )
        })
}

fn get_rustc_src(sysroot_path: &AbsPath) -> Option<ManifestPath> {
    let rustc_src = sysroot_path.join("lib/rustlib/rustc-src/rust/compiler/rustc/Cargo.toml");
    let rustc_src = ManifestPath::try_from(rustc_src).ok()?;
    tracing::debug!("checking for rustc source code: {}", rustc_src.display());
    if fs::metadata(&rustc_src).is_ok() {
        Some(rustc_src)
    } else {
        None
    }
}

fn get_rust_src(sysroot_path: &AbsPath) -> Option<AbsPathBuf> {
    let rust_src = sysroot_path.join("lib/rustlib/src/rust/library");
    tracing::debug!("checking sysroot library: {}", rust_src.display());
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
