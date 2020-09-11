//! Loads "sysroot" crate.
//!
//! One confusing point here is that normally sysroot is a bunch of `.rlib`s,
//! but we can't process `.rlib` and need source code instead. The source code
//! is typically installed with `rustup component add rust-src` command.

use std::{convert::TryFrom, env, ops, path::PathBuf, process::Command};

use anyhow::{format_err, Result};
use arena::{Arena, Idx};
use paths::{AbsPath, AbsPathBuf};

use crate::utf8_stdout;

#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct Sysroot {
    crates: Arena<SysrootCrateData>,
}

pub type SysrootCrate = Idx<SysrootCrateData>;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SysrootCrateData {
    pub name: String,
    pub root: AbsPathBuf,
    pub deps: Vec<SysrootCrate>,
}

impl ops::Index<SysrootCrate> for Sysroot {
    type Output = SysrootCrateData;
    fn index(&self, index: SysrootCrate) -> &SysrootCrateData {
        &self.crates[index]
    }
}

impl Sysroot {
    pub fn public_deps(&self) -> impl Iterator<Item = (&'static str, SysrootCrate)> + '_ {
        // core is added as a dependency before std in order to
        // mimic rustcs dependency order
        vec!["core", "alloc", "std"].into_iter().filter_map(move |it| Some((it, self.by_name(it)?)))
    }

    pub fn proc_macro(&self) -> Option<SysrootCrate> {
        self.by_name("proc_macro")
    }

    pub fn crates<'a>(&'a self) -> impl Iterator<Item = SysrootCrate> + ExactSizeIterator + 'a {
        self.crates.iter().map(|(id, _data)| id)
    }

    pub fn discover(cargo_toml: &AbsPath) -> Result<Sysroot> {
        let current_dir = cargo_toml.parent().unwrap();
        let sysroot_src_dir = discover_sysroot_src_dir(current_dir)?;
        let res = Sysroot::load(&sysroot_src_dir)?;
        Ok(res)
    }

    pub fn load(sysroot_src_dir: &AbsPath) -> Result<Sysroot> {
        let mut sysroot = Sysroot { crates: Arena::default() };

        for name in SYSROOT_CRATES.trim().lines() {
            // FIXME: first path when 1.47 comes out
            // https://github.com/rust-lang/rust/pull/73265
            let root = [format!("lib{}/lib.rs", name), format!("{}/src/lib.rs", name)]
                .iter()
                .map(|it| sysroot_src_dir.join(it))
                .find(|it| it.exists());

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
            if let Some(core) = sysroot.by_name("core") {
                sysroot.crates[alloc].deps.push(core);
            }
        }

        if sysroot.by_name("core").is_none() {
            anyhow::bail!(
                "could not find libcore in sysroot path `{}`",
                sysroot_src_dir.as_ref().display()
            );
        }

        Ok(sysroot)
    }

    fn by_name(&self, name: &str) -> Option<SysrootCrate> {
        let (id, _data) = self.crates.iter().find(|(_id, data)| data.name == name)?;
        Some(id)
    }
}

fn discover_sysroot_src_dir(current_dir: &AbsPath) -> Result<AbsPathBuf> {
    if let Ok(path) = env::var("RUST_SRC_PATH") {
        let path = AbsPathBuf::try_from(path.as_str())
            .map_err(|path| format_err!("RUST_SRC_PATH must be absolute: {}", path.display()))?;
        return Ok(path);
    }

    let sysroot_path = {
        let mut rustc = Command::new(toolchain::rustc());
        rustc.current_dir(current_dir).args(&["--print", "sysroot"]);
        let stdout = utf8_stdout(rustc)?;
        AbsPathBuf::assert(PathBuf::from(stdout))
    };

    get_rust_src(&sysroot_path)
        .or_else(|| {
            let mut rustup = Command::new(toolchain::rustup());
            rustup.current_dir(current_dir).args(&["component", "add", "rust-src"]);
            utf8_stdout(rustup).ok()?;
            get_rust_src(&sysroot_path)
        })
        .ok_or_else(|| {
            format_err!(
                "\
can't load standard library from sysroot
{}
(discovered via `rustc --print sysroot`)
try running `rustup component add rust-src` or set `RUST_SRC_PATH`",
                sysroot_path.display(),
            )
        })
}

fn get_rust_src(sysroot_path: &AbsPath) -> Option<AbsPathBuf> {
    // Try the new path first since the old one still exists.
    //
    // FIXME: remove `src` when 1.47 comes out
    // https://github.com/rust-lang/rust/pull/73265
    let rust_src = sysroot_path.join("lib/rustlib/src/rust");
    ["library", "src"].iter().map(|it| rust_src.join(it)).find(|it| it.exists())
}

impl SysrootCrateData {
    pub fn root_dir(&self) -> &AbsPath {
        self.root.parent().unwrap()
    }
}

const SYSROOT_CRATES: &str = "
alloc
core
panic_abort
panic_unwind
proc_macro
profiler_builtins
rtstartup
std
stdarch
term
test
unwind";

const STD_DEPS: &str = "
alloc
core
panic_abort
panic_unwind
profiler_builtins
rtstartup
proc_macro
stdarch
term
test
unwind";
