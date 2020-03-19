//! FIXME: write short doc here

use anyhow::{bail, Context, Result};
use std::{
    env, ops,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use ra_arena::{impl_arena_id, Arena, RawId};

#[derive(Default, Debug, Clone)]
pub struct Sysroot {
    crates: Arena<SysrootCrate, SysrootCrateData>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SysrootCrate(RawId);
impl_arena_id!(SysrootCrate);

#[derive(Debug, Clone)]
pub struct SysrootCrateData {
    pub name: String,
    pub root: PathBuf,
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

    pub fn discover(cargo_toml: &Path) -> Result<Sysroot> {
        let src = get_or_install_rust_src(cargo_toml)?;
        let mut sysroot = Sysroot { crates: Arena::default() };
        for name in SYSROOT_CRATES.trim().lines() {
            let root = src.join(format!("lib{}", name)).join("lib.rs");
            if root.exists() {
                sysroot.crates.alloc(SysrootCrateData {
                    name: name.into(),
                    root,
                    deps: Vec::new(),
                });
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

fn create_command_text(program: &str, args: &[&str]) -> String {
    format!("{} {}", program, args.join(" "))
}

fn run_command_in_cargo_dir(cargo_toml: &Path, program: &str, args: &[&str]) -> Result<Output> {
    let output = Command::new(program)
        .current_dir(cargo_toml.parent().unwrap())
        .args(args)
        .output()
        .context(format!("{} failed", create_command_text(program, args)))?;
    if !output.status.success() {
        match output.status.code() {
            Some(code) => bail!(
                "failed to run the command: '{}' exited with code {}",
                create_command_text(program, args),
                code
            ),
            None => bail!(
                "failed to run the command: '{}' terminated by signal",
                create_command_text(program, args)
            ),
        };
    }
    Ok(output)
}

fn get_or_install_rust_src(cargo_toml: &Path) -> Result<PathBuf> {
    if let Ok(path) = env::var("RUST_SRC_PATH") {
        return Ok(path.into());
    }
    let rustc_output = run_command_in_cargo_dir(cargo_toml, "rustc", &["--print", "sysroot"])?;
    let stdout = String::from_utf8(rustc_output.stdout)?;
    let sysroot_path = Path::new(stdout.trim());
    let src_path = sysroot_path.join("lib/rustlib/src/rust/src");

    if !src_path.exists() {
        run_command_in_cargo_dir(cargo_toml, "rustup", &["component", "add", "rust-src"])?;
    }
    if !src_path.exists() {
        bail!(
            "can't load standard library from sysroot\n\
            {}\n\
            (discovered via `rustc --print sysroot`)\n\
            try running `rustup component add rust-src` or set `RUST_SRC_PATH`",
            src_path.display(),
        )
    }
    Ok(src_path)
}

impl SysrootCrateData {
    pub fn root_dir(&self) -> &Path {
        self.root.parent().unwrap()
    }
}

const SYSROOT_CRATES: &str = "
std
core
alloc
collections
libc
panic_unwind
proc_macro
rustc_unicode
std_unicode
test
alloc_jemalloc
alloc_system
compiler_builtins
getopts
panic_unwind
panic_abort
rand
term
unwind
build_helper
rustc_asan
rustc_lsan
rustc_msan
rustc_tsan
syntax";

const STD_DEPS: &str = "
alloc
alloc_jemalloc
alloc_system
core
panic_abort
rand
compiler_builtins
unwind
rustc_asan
rustc_lsan
rustc_msan
rustc_tsan
build_helper";
