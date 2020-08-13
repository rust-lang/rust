//! FIXME: write short doc here

use std::{convert::TryFrom, env, ops, path::Path, process::Command};

use anyhow::{bail, format_err, Result};
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
    pub fn core(&self) -> Option<SysrootCrate> {
        self.by_name("core")
    }

    pub fn alloc(&self) -> Option<SysrootCrate> {
        self.by_name("alloc")
    }

    pub fn std(&self) -> Option<SysrootCrate> {
        self.by_name("std")
    }

    pub fn proc_macro(&self) -> Option<SysrootCrate> {
        self.by_name("proc_macro")
    }

    pub fn crates<'a>(&'a self) -> impl Iterator<Item = SysrootCrate> + ExactSizeIterator + 'a {
        self.crates.iter().map(|(id, _data)| id)
    }

    pub fn discover(cargo_toml: &AbsPath) -> Result<Sysroot> {
        let src = get_or_install_rust_src(cargo_toml)?;
        let mut sysroot = Sysroot { crates: Arena::default() };
        for name in SYSROOT_CRATES.trim().lines() {
            // FIXME: remove this path when 1.47 comes out
            // https://github.com/rust-lang/rust/pull/73265
            let root = src.join(format!("lib{}", name)).join("lib.rs");
            if root.exists() {
                sysroot.crates.alloc(SysrootCrateData {
                    name: name.into(),
                    root,
                    deps: Vec::new(),
                });
            } else {
                let root = src.join(name).join("src/lib.rs");
                if root.exists() {
                    sysroot.crates.alloc(SysrootCrateData {
                        name: name.into(),
                        root,
                        deps: Vec::new(),
                    });
                }
            }
        }
        if let Some(std) = sysroot.std() {
            for dep in STD_DEPS.trim().lines() {
                if let Some(dep) = sysroot.by_name(dep) {
                    sysroot.crates[std].deps.push(dep)
                }
            }
        }
        if let Some(alloc) = sysroot.alloc() {
            if let Some(core) = sysroot.core() {
                sysroot.crates[alloc].deps.push(core);
            }
        }
        Ok(sysroot)
    }

    fn by_name(&self, name: &str) -> Option<SysrootCrate> {
        self.crates.iter().find(|(_id, data)| data.name == name).map(|(id, _data)| id)
    }
}

fn get_or_install_rust_src(cargo_toml: &AbsPath) -> Result<AbsPathBuf> {
    if let Ok(path) = env::var("RUST_SRC_PATH") {
        let path = AbsPathBuf::try_from(path.as_str())
            .map_err(|path| format_err!("RUST_SRC_PATH must be absolute: {}", path.display()))?;
        return Ok(path);
    }
    let current_dir = cargo_toml.parent().unwrap();
    let mut rustc = Command::new(toolchain::rustc());
    rustc.current_dir(current_dir).args(&["--print", "sysroot"]);
    let stdout = utf8_stdout(rustc)?;
    let sysroot_path = AbsPath::assert(Path::new(stdout.trim()));
    let mut src = get_rust_src(sysroot_path);
    if src.is_none() {
        let mut rustup = Command::new(toolchain::rustup());
        rustup.current_dir(current_dir).args(&["component", "add", "rust-src"]);
        utf8_stdout(rustup)?;
        src = get_rust_src(sysroot_path);
    }
    match src {
        Some(r) => Ok(r),
        None => bail!(
            "can't load standard library from sysroot\n\
            {}\n\
            (discovered via `rustc --print sysroot`)\n\
            try running `rustup component add rust-src` or set `RUST_SRC_PATH`",
            sysroot_path.display(),
        ),
    }
}

fn get_rust_src(sysroot_path: &AbsPath) -> Option<AbsPathBuf> {
    // try the new path first since the old one still exists
    let mut src_path = sysroot_path.join("lib/rustlib/src/rust/library");
    if !src_path.exists() {
        // FIXME: remove this path when 1.47 comes out
        // https://github.com/rust-lang/rust/pull/73265
        src_path = sysroot_path.join("lib/rustlib/src/rust/src");
    }
    if src_path.exists() {
        Some(src_path)
    } else {
        None
    }
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
