//! Loads "sysroot" crate.
//!
//! One confusing point here is that normally sysroot is a bunch of `.rlib`s,
//! but we can't process `.rlib` and need source code instead. The source code
//! is typically installed with `rustup component add rust-src` command.

use std::{env, fs, ops, process::Command};

use anyhow::{format_err, Result};
use base_db::CrateName;
use itertools::Itertools;
use la_arena::{Arena, Idx};
use paths::{AbsPath, AbsPathBuf, Utf8PathBuf};
use rustc_hash::FxHashMap;
use toolchain::{probe_for_binary, Tool};

use crate::{utf8_stdout, CargoConfig, CargoWorkspace, ManifestPath, SysrootQueryMetadata};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sysroot {
    root: Option<AbsPathBuf>,
    src_root: Option<AbsPathBuf>,
    mode: SysrootMode,
    error: Option<String>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum SysrootMode {
    Workspace(CargoWorkspace),
    Stitched(Stitched),
    Empty,
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
        [("core", true), ("alloc", false), ("std", true), ("test", false)].into_iter().filter_map(
            move |(name, prelude)| {
                Some((CrateName::new(name).unwrap(), self.by_name(name)?, prelude))
            },
        )
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
    pub const fn empty() -> Sysroot {
        Sysroot { root: None, src_root: None, mode: SysrootMode::Empty, error: None }
    }

    /// Returns sysroot "root" directory, where `bin/`, `etc/`, `lib/`, `libexec/`
    /// subfolder live, like:
    /// `$HOME/.rustup/toolchains/nightly-2022-07-23-x86_64-unknown-linux-gnu`
    pub fn root(&self) -> Option<&AbsPath> {
        self.root.as_deref()
    }

    /// Returns the sysroot "source" directory, where stdlib sources are located, like:
    /// `$HOME/.rustup/toolchains/nightly-2022-07-23-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library`
    pub fn src_root(&self) -> Option<&AbsPath> {
        self.src_root.as_deref()
    }

    pub fn is_empty(&self) -> bool {
        match &self.mode {
            SysrootMode::Workspace(ws) => ws.packages().next().is_none(),
            SysrootMode::Stitched(stitched) => stitched.crates.is_empty(),
            SysrootMode::Empty => true,
        }
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn num_packages(&self) -> usize {
        match &self.mode {
            SysrootMode::Workspace(ws) => ws.packages().count(),
            SysrootMode::Stitched(c) => c.crates().count(),
            SysrootMode::Empty => 0,
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
        sysroot_query_metadata: SysrootQueryMetadata,
    ) -> Sysroot {
        let sysroot_dir = discover_sysroot_dir(dir, extra_env);
        let sysroot_src_dir = sysroot_dir.as_ref().ok().map(|sysroot_dir| {
            discover_sysroot_src_dir_or_add_component(sysroot_dir, dir, extra_env)
        });
        Sysroot::load_core_check(Some(sysroot_dir), sysroot_src_dir, sysroot_query_metadata)
    }

    pub fn discover_with_src_override(
        current_dir: &AbsPath,
        extra_env: &FxHashMap<String, String>,
        sysroot_src_dir: AbsPathBuf,
        sysroot_query_metadata: SysrootQueryMetadata,
    ) -> Sysroot {
        let sysroot_dir = discover_sysroot_dir(current_dir, extra_env);
        Sysroot::load_core_check(
            Some(sysroot_dir),
            Some(Ok(sysroot_src_dir)),
            sysroot_query_metadata,
        )
    }

    pub fn discover_sysroot_src_dir(
        sysroot_dir: AbsPathBuf,
        sysroot_query_metadata: SysrootQueryMetadata,
    ) -> Sysroot {
        let sysroot_src_dir = discover_sysroot_src_dir(&sysroot_dir)
            .ok_or_else(|| format_err!("can't find standard library sources in {sysroot_dir}"));
        Sysroot::load_core_check(
            Some(Ok(sysroot_dir)),
            Some(sysroot_src_dir),
            sysroot_query_metadata,
        )
    }

    pub fn discover_rustc_src(&self) -> Option<ManifestPath> {
        get_rustc_src(self.root()?)
    }

    /// Returns a command to run a tool preferring the cargo proxies if the sysroot exists.
    pub fn tool(&self, tool: Tool) -> Command {
        match self.root() {
            Some(root) => {
                // special case rustc, we can look that up directly in the sysroot's bin folder
                // as it should never invoke another cargo binary
                if let Tool::Rustc = tool {
                    if let Some(path) =
                        probe_for_binary(root.join("bin").join(Tool::Rustc.name()).into())
                    {
                        return Command::new(path);
                    }
                }

                let mut cmd = Command::new(tool.prefer_proxy());
                cmd.env("RUSTUP_TOOLCHAIN", AsRef::<std::path::Path>::as_ref(root));
                cmd
            }
            _ => Command::new(tool.path()),
        }
    }

    pub fn discover_proc_macro_srv(&self) -> anyhow::Result<AbsPathBuf> {
        let Some(root) = self.root() else {
            return Err(anyhow::format_err!("no sysroot",));
        };
        ["libexec", "lib"]
            .into_iter()
            .map(|segment| root.join(segment).join("rust-analyzer-proc-macro-srv"))
            .find_map(|server_path| probe_for_binary(server_path.into()))
            .map(AbsPathBuf::assert)
            .ok_or_else(|| {
                anyhow::format_err!("cannot find proc-macro server in sysroot `{}`", root)
            })
    }

    pub fn load(
        sysroot_dir: Option<AbsPathBuf>,
        sysroot_src_dir: Option<AbsPathBuf>,
        sysroot_query_metadata: SysrootQueryMetadata,
    ) -> Sysroot {
        Self::load_core_check(sysroot_dir.map(Ok), sysroot_src_dir.map(Ok), sysroot_query_metadata)
    }

    fn load_core_check(
        sysroot_dir: Option<Result<AbsPathBuf, anyhow::Error>>,
        sysroot_src_dir: Option<Result<AbsPathBuf, anyhow::Error>>,
        sysroot_query_metadata: SysrootQueryMetadata,
    ) -> Sysroot {
        let mut sysroot = Self::load_(sysroot_dir, sysroot_src_dir, sysroot_query_metadata);
        if sysroot.error.is_none() {
            if let Some(src_root) = &sysroot.src_root {
                let has_core = match &sysroot.mode {
                    SysrootMode::Workspace(ws) => ws.packages().any(|p| ws[p].name == "core"),
                    SysrootMode::Stitched(stitched) => stitched.by_name("core").is_some(),
                    SysrootMode::Empty => true,
                };
                if !has_core {
                    let var_note = if env::var_os("RUST_SRC_PATH").is_some() {
                        " (env var `RUST_SRC_PATH` is set and may be incorrect, try unsetting it)"
                    } else {
                        ", try running `rustup component add rust-src` to possibly fix this"
                    };
                    sysroot.error = Some(format!(
                        "sysroot at `{src_root}` is missing a `core` library{var_note}",
                    ));
                }
            }
        }
        sysroot
    }

    fn load_(
        sysroot_dir: Option<Result<AbsPathBuf, anyhow::Error>>,
        sysroot_src_dir: Option<Result<AbsPathBuf, anyhow::Error>>,
        sysroot_query_metadata: SysrootQueryMetadata,
    ) -> Sysroot {
        let sysroot_dir = match sysroot_dir {
            Some(Ok(sysroot_dir)) => Some(sysroot_dir),
            Some(Err(e)) => {
                return Sysroot {
                    root: None,
                    src_root: None,
                    mode: SysrootMode::Empty,
                    error: Some(e.to_string()),
                }
            }
            None => None,
        };
        let sysroot_src_dir = match sysroot_src_dir {
            Some(Ok(sysroot_src_dir)) => sysroot_src_dir,
            Some(Err(e)) => {
                return Sysroot {
                    root: sysroot_dir,
                    src_root: None,
                    mode: SysrootMode::Empty,
                    error: Some(e.to_string()),
                }
            }
            None => {
                return Sysroot {
                    root: sysroot_dir,
                    src_root: None,
                    mode: SysrootMode::Empty,
                    error: None,
                }
            }
        };
        if sysroot_query_metadata == SysrootQueryMetadata::CargoMetadata {
            let library_manifest =
                ManifestPath::try_from(sysroot_src_dir.join("Cargo.toml")).unwrap();
            if fs::metadata(&library_manifest).is_ok() {
                if let Some(sysroot) =
                    Self::load_library_via_cargo(library_manifest, &sysroot_dir, &sysroot_src_dir)
                {
                    return sysroot;
                }
            }
        }
        tracing::debug!("Stitching sysroot library: {sysroot_src_dir}");

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
            src_root: Some(sysroot_src_dir),
            mode: SysrootMode::Stitched(stitched),
            error: None,
        }
    }

    fn load_library_via_cargo(
        library_manifest: ManifestPath,
        sysroot_dir: &Option<AbsPathBuf>,
        sysroot_src_dir: &AbsPathBuf,
    ) -> Option<Sysroot> {
        tracing::debug!("Loading library metadata: {library_manifest}");
        let mut cargo_config = CargoConfig::default();
        // the sysroot uses `public-dependency`, so we make cargo think it's a nightly
        cargo_config.extra_env.insert(
            "__CARGO_TEST_CHANNEL_OVERRIDE_DO_NOT_USE_THIS".to_owned(),
            "nightly".to_owned(),
        );

        let (mut res, _) = match CargoWorkspace::fetch_metadata(
            &library_manifest,
            sysroot_src_dir,
            &cargo_config,
            &Sysroot::empty(),
            // Make sure we never attempt to write to the sysroot
            true,
            &|_| (),
        ) {
            Ok(it) => it,
            Err(e) => {
                tracing::error!("`cargo metadata` failed on `{library_manifest}` : {e}");
                return None;
            }
        };

        // Patch out `rustc-std-workspace-*` crates to point to the real crates.
        // This is done prior to `CrateGraph` construction to prevent de-duplication logic from failing.
        let patches = {
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

            [fake_core.zip(real_core), fake_alloc.zip(real_alloc), fake_std.zip(real_std)]
                .into_iter()
                .flatten()
        };

        if let Some(resolve) = res.resolve.as_mut() {
            resolve.nodes.retain_mut(|node| {
                // Replace `rustc-std-workspace` crate with the actual one in the dependency list
                node.deps.iter_mut().for_each(|dep| {
                    let real_pkg = patches.clone().find(|((_, fake_id), _)| *fake_id == dep.pkg);
                    if let Some((_, real)) = real_pkg {
                        dep.pkg = real;
                    }
                });
                // Remove this node if it's a fake one
                !patches.clone().any(|((_, fake), _)| fake == node.id)
            });
        }
        // Remove the fake ones from the package list
        patches.map(|((idx, _), _)| idx).sorted().rev().for_each(|idx| {
            res.packages.remove(idx);
        });

        let cargo_workspace = CargoWorkspace::new(res, library_manifest);
        Some(Sysroot {
            root: sysroot_dir.clone(),
            src_root: Some(sysroot_src_dir.clone()),
            mode: SysrootMode::Workspace(cargo_workspace),
            error: None,
        })
    }
}

fn discover_sysroot_dir(
    current_dir: &AbsPath,
    extra_env: &FxHashMap<String, String>,
) -> Result<AbsPathBuf> {
    let mut rustc = Command::new(Tool::Rustc.path());
    rustc.envs(extra_env);
    rustc.current_dir(current_dir).args(["--print", "sysroot"]);
    tracing::debug!("Discovering sysroot by {:?}", rustc);
    let stdout = utf8_stdout(rustc)?;
    Ok(AbsPathBuf::assert(Utf8PathBuf::from(stdout)))
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
            let mut rustup = Command::new(Tool::Rustup.prefer_proxy());
            rustup.envs(extra_env);
            rustup.current_dir(current_dir).args(["component", "add", "rust-src"]);
            tracing::info!("adding rust-src component by {:?}", rustup);
            utf8_stdout(rustup).ok()?;
            get_rust_src(sysroot_path)
        })
        .ok_or_else(|| {
            tracing::error!(%sysroot_path, "can't load standard library, try installing `rust-src`");
            format_err!(
                "\
can't load standard library from sysroot
{sysroot_path}
(discovered via `rustc --print sysroot`)
try installing `rust-src` the same way you installed `rustc`"
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
