//! Loads "sysroot" crate.
//!
//! One confusing point here is that normally sysroot is a bunch of `.rlib`s,
//! but we can't process `.rlib` and need source code instead. The source code
//! is typically installed with `rustup component add rust-src` command.

use std::{env, fs, ops::Not, path::Path, process::Command};

use anyhow::{Result, format_err};
use itertools::Itertools;
use paths::{AbsPath, AbsPathBuf, Utf8PathBuf};
use rustc_hash::FxHashMap;
use stdx::format_to;
use toolchain::{Tool, probe_for_binary};

use crate::{
    CargoWorkspace, ManifestPath, ProjectJson, RustSourceWorkspaceConfig,
    cargo_workspace::CargoMetadataConfig, utf8_stdout,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sysroot {
    root: Option<AbsPathBuf>,
    rust_lib_src_root: Option<AbsPathBuf>,
    workspace: RustLibSrcWorkspace,
    error: Option<String>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum RustLibSrcWorkspace {
    Workspace(CargoWorkspace),
    Json(ProjectJson),
    Stitched(stitched::Stitched),
    Empty,
}

impl Sysroot {
    pub const fn empty() -> Sysroot {
        Sysroot {
            root: None,
            rust_lib_src_root: None,
            workspace: RustLibSrcWorkspace::Empty,
            error: None,
        }
    }

    /// Returns sysroot "root" directory, where `bin/`, `etc/`, `lib/`, `libexec/`
    /// subfolder live, like:
    /// `$HOME/.rustup/toolchains/nightly-2022-07-23-x86_64-unknown-linux-gnu`
    pub fn root(&self) -> Option<&AbsPath> {
        self.root.as_deref()
    }

    /// Returns the sysroot "source" directory, where stdlib sources are located, like:
    /// `$HOME/.rustup/toolchains/nightly-2022-07-23-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library`
    pub fn rust_lib_src_root(&self) -> Option<&AbsPath> {
        self.rust_lib_src_root.as_deref()
    }

    pub fn is_rust_lib_src_empty(&self) -> bool {
        match &self.workspace {
            RustLibSrcWorkspace::Workspace(ws) => ws.packages().next().is_none(),
            RustLibSrcWorkspace::Json(project_json) => project_json.n_crates() == 0,
            RustLibSrcWorkspace::Stitched(stitched) => stitched.crates.is_empty(),
            RustLibSrcWorkspace::Empty => true,
        }
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn num_packages(&self) -> usize {
        match &self.workspace {
            RustLibSrcWorkspace::Workspace(ws) => ws.packages().count(),
            RustLibSrcWorkspace::Json(project_json) => project_json.n_crates(),
            RustLibSrcWorkspace::Stitched(stitched) => stitched.crates.len(),
            RustLibSrcWorkspace::Empty => 0,
        }
    }

    pub(crate) fn workspace(&self) -> &RustLibSrcWorkspace {
        &self.workspace
    }
}

impl Sysroot {
    /// Attempts to discover the toolchain's sysroot from the given `dir`.
    pub fn discover(dir: &AbsPath, extra_env: &FxHashMap<String, Option<String>>) -> Sysroot {
        let sysroot_dir = discover_sysroot_dir(dir, extra_env);
        let rust_lib_src_dir = sysroot_dir.as_ref().ok().map(|sysroot_dir| {
            discover_rust_lib_src_dir_or_add_component(sysroot_dir, dir, extra_env)
        });
        Sysroot::assemble(Some(sysroot_dir), rust_lib_src_dir)
    }

    pub fn discover_with_src_override(
        current_dir: &AbsPath,
        extra_env: &FxHashMap<String, Option<String>>,
        rust_lib_src_dir: AbsPathBuf,
    ) -> Sysroot {
        let sysroot_dir = discover_sysroot_dir(current_dir, extra_env);
        Sysroot::assemble(Some(sysroot_dir), Some(Ok(rust_lib_src_dir)))
    }

    pub fn discover_rust_lib_src_dir(sysroot_dir: AbsPathBuf) -> Sysroot {
        let rust_lib_src_dir = discover_rust_lib_src_dir(&sysroot_dir)
            .ok_or_else(|| format_err!("can't find standard library sources in {sysroot_dir}"));
        Sysroot::assemble(Some(Ok(sysroot_dir)), Some(rust_lib_src_dir))
    }

    pub fn discover_rustc_src(&self) -> Option<ManifestPath> {
        get_rustc_src(self.root()?)
    }

    pub fn new(sysroot_dir: Option<AbsPathBuf>, rust_lib_src_dir: Option<AbsPathBuf>) -> Sysroot {
        Self::assemble(sysroot_dir.map(Ok), rust_lib_src_dir.map(Ok))
    }

    /// Returns a command to run a tool preferring the cargo proxies if the sysroot exists.
    pub fn tool(
        &self,
        tool: Tool,
        current_dir: impl AsRef<Path>,
        envs: &FxHashMap<String, Option<String>>,
    ) -> Command {
        match self.root() {
            Some(root) => {
                // special case rustc, we can look that up directly in the sysroot's bin folder
                // as it should never invoke another cargo binary
                if let Tool::Rustc = tool {
                    if let Some(path) =
                        probe_for_binary(root.join("bin").join(Tool::Rustc.name()).into())
                    {
                        return toolchain::command(path, current_dir, envs);
                    }
                }

                let mut cmd = toolchain::command(tool.prefer_proxy(), current_dir, envs);
                if !envs.contains_key("RUSTUP_TOOLCHAIN")
                    && std::env::var_os("RUSTUP_TOOLCHAIN").is_none()
                {
                    cmd.env("RUSTUP_TOOLCHAIN", AsRef::<std::path::Path>::as_ref(root));
                }

                cmd
            }
            _ => toolchain::command(tool.path(), current_dir, envs),
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

    fn assemble(
        sysroot_dir: Option<Result<AbsPathBuf, anyhow::Error>>,
        rust_lib_src_dir: Option<Result<AbsPathBuf, anyhow::Error>>,
    ) -> Sysroot {
        let mut errors = String::new();
        let root = match sysroot_dir {
            Some(Ok(sysroot_dir)) => Some(sysroot_dir),
            Some(Err(e)) => {
                format_to!(errors, "{e}\n");
                None
            }
            None => None,
        };
        let rust_lib_src_root = match rust_lib_src_dir {
            Some(Ok(rust_lib_src_dir)) => Some(rust_lib_src_dir),
            Some(Err(e)) => {
                format_to!(errors, "{e}\n");
                None
            }
            None => None,
        };
        Sysroot {
            root,
            rust_lib_src_root,
            workspace: RustLibSrcWorkspace::Empty,
            error: errors.is_empty().not().then_some(errors),
        }
    }

    pub fn load_workspace(
        &self,
        sysroot_source_config: &RustSourceWorkspaceConfig,
    ) -> Option<RustLibSrcWorkspace> {
        assert!(matches!(self.workspace, RustLibSrcWorkspace::Empty), "workspace already loaded");
        let Self { root: _, rust_lib_src_root: Some(src_root), workspace: _, error: _ } = self
        else {
            return None;
        };
        if let RustSourceWorkspaceConfig::CargoMetadata(cargo_config) = sysroot_source_config {
            let library_manifest = ManifestPath::try_from(src_root.join("Cargo.toml")).unwrap();
            if fs::metadata(&library_manifest).is_ok() {
                if let Some(loaded) =
                    self.load_library_via_cargo(library_manifest, src_root, cargo_config)
                {
                    return Some(loaded);
                }
            }
            tracing::debug!("Stitching sysroot library: {src_root}");

            let mut stitched = stitched::Stitched { crates: Default::default() };

            for path in stitched::SYSROOT_CRATES.trim().lines() {
                let name = path.split('/').next_back().unwrap();
                let root = [format!("{path}/src/lib.rs"), format!("lib{path}/lib.rs")]
                    .into_iter()
                    .map(|it| src_root.join(it))
                    .filter_map(|it| ManifestPath::try_from(it).ok())
                    .find(|it| fs::metadata(it).is_ok());

                if let Some(root) = root {
                    stitched.crates.alloc(stitched::RustLibSrcCrateData {
                        name: name.into(),
                        root,
                        deps: Vec::new(),
                    });
                }
            }

            if let Some(std) = stitched.by_name("std") {
                for dep in stitched::STD_DEPS.trim().lines() {
                    if let Some(dep) = stitched.by_name(dep) {
                        stitched.crates[std].deps.push(dep)
                    }
                }
            }

            if let Some(alloc) = stitched.by_name("alloc") {
                for dep in stitched::ALLOC_DEPS.trim().lines() {
                    if let Some(dep) = stitched.by_name(dep) {
                        stitched.crates[alloc].deps.push(dep)
                    }
                }
            }

            if let Some(proc_macro) = stitched.by_name("proc_macro") {
                for dep in stitched::PROC_MACRO_DEPS.trim().lines() {
                    if let Some(dep) = stitched.by_name(dep) {
                        stitched.crates[proc_macro].deps.push(dep)
                    }
                }
            }
            return Some(RustLibSrcWorkspace::Stitched(stitched));
        } else if let RustSourceWorkspaceConfig::Json(project_json) = sysroot_source_config {
            return Some(RustLibSrcWorkspace::Json(project_json.clone()));
        }

        None
    }

    pub fn set_workspace(&mut self, workspace: RustLibSrcWorkspace) {
        self.workspace = workspace;
        if self.error.is_none() {
            if let Some(src_root) = &self.rust_lib_src_root {
                let has_core = match &self.workspace {
                    RustLibSrcWorkspace::Workspace(ws) => {
                        ws.packages().any(|p| ws[p].name == "core")
                    }
                    RustLibSrcWorkspace::Json(project_json) => project_json
                        .crates()
                        .filter_map(|(_, krate)| krate.display_name.clone())
                        .any(|name| name.canonical_name().as_str() == "core"),
                    RustLibSrcWorkspace::Stitched(stitched) => stitched.by_name("core").is_some(),
                    RustLibSrcWorkspace::Empty => true,
                };
                if !has_core {
                    let var_note = if env::var_os("RUST_SRC_PATH").is_some() {
                        " (env var `RUST_SRC_PATH` is set and may be incorrect, try unsetting it)"
                    } else {
                        ", try running `rustup component add rust-src` to possibly fix this"
                    };
                    self.error = Some(format!(
                        "sysroot at `{src_root}` is missing a `core` library{var_note}",
                    ));
                }
            }
        }
    }

    fn load_library_via_cargo(
        &self,
        library_manifest: ManifestPath,
        rust_lib_src_dir: &AbsPathBuf,
        cargo_config: &CargoMetadataConfig,
    ) -> Option<RustLibSrcWorkspace> {
        tracing::debug!("Loading library metadata: {library_manifest}");
        let mut cargo_config = cargo_config.clone();
        // the sysroot uses `public-dependency`, so we make cargo think it's a nightly
        cargo_config.extra_env.insert(
            "__CARGO_TEST_CHANNEL_OVERRIDE_DO_NOT_USE_THIS".to_owned(),
            Some("nightly".to_owned()),
        );

        let (mut res, _) = match CargoWorkspace::fetch_metadata(
            &library_manifest,
            rust_lib_src_dir,
            &cargo_config,
            self,
            false,
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
                    None => match &**package.name {
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

        let cargo_workspace = CargoWorkspace::new(res, library_manifest, Default::default(), true);
        Some(RustLibSrcWorkspace::Workspace(cargo_workspace))
    }
}

fn discover_sysroot_dir(
    current_dir: &AbsPath,
    extra_env: &FxHashMap<String, Option<String>>,
) -> Result<AbsPathBuf> {
    let mut rustc = toolchain::command(Tool::Rustc.path(), current_dir, extra_env);
    rustc.current_dir(current_dir).args(["--print", "sysroot"]);
    tracing::debug!("Discovering sysroot by {:?}", rustc);
    let stdout = utf8_stdout(&mut rustc)?;
    Ok(AbsPathBuf::assert(Utf8PathBuf::from(stdout)))
}

fn discover_rust_lib_src_dir(sysroot_path: &AbsPathBuf) -> Option<AbsPathBuf> {
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

    get_rust_lib_src(sysroot_path)
}

fn discover_rust_lib_src_dir_or_add_component(
    sysroot_path: &AbsPathBuf,
    current_dir: &AbsPath,
    extra_env: &FxHashMap<String, Option<String>>,
) -> Result<AbsPathBuf> {
    discover_rust_lib_src_dir(sysroot_path)
        .or_else(|| {
            let mut rustup = toolchain::command(Tool::Rustup.prefer_proxy(), current_dir, extra_env);
            rustup.args(["component", "add", "rust-src"]);
            tracing::info!("adding rust-src component by {:?}", rustup);
            utf8_stdout(&mut rustup).ok()?;
            get_rust_lib_src(sysroot_path)
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
    if fs::metadata(&rustc_src).is_ok() { Some(rustc_src) } else { None }
}

fn get_rust_lib_src(sysroot_path: &AbsPath) -> Option<AbsPathBuf> {
    let rust_lib_src = sysroot_path.join("lib/rustlib/src/rust/library");
    tracing::debug!("checking sysroot library: {rust_lib_src}");
    if fs::metadata(&rust_lib_src).is_ok() { Some(rust_lib_src) } else { None }
}

// FIXME: Remove this, that will bump our project MSRV to 1.82
pub(crate) mod stitched {
    use std::ops;

    use base_db::CrateName;
    use la_arena::{Arena, Idx};

    use crate::ManifestPath;

    #[derive(Debug, Clone, Eq, PartialEq)]
    pub struct Stitched {
        pub(super) crates: Arena<RustLibSrcCrateData>,
    }

    impl ops::Index<RustLibSrcCrate> for Stitched {
        type Output = RustLibSrcCrateData;
        fn index(&self, index: RustLibSrcCrate) -> &RustLibSrcCrateData {
            &self.crates[index]
        }
    }

    impl Stitched {
        pub(crate) fn public_deps(
            &self,
        ) -> impl Iterator<Item = (CrateName, RustLibSrcCrate, bool)> + '_ {
            // core is added as a dependency before std in order to
            // mimic rustcs dependency order
            [("core", true), ("alloc", false), ("std", true), ("test", false)]
                .into_iter()
                .filter_map(move |(name, prelude)| {
                    Some((CrateName::new(name).unwrap(), self.by_name(name)?, prelude))
                })
        }

        pub(crate) fn proc_macro(&self) -> Option<RustLibSrcCrate> {
            self.by_name("proc_macro")
        }

        pub(crate) fn crates(&self) -> impl ExactSizeIterator<Item = RustLibSrcCrate> + '_ {
            self.crates.iter().map(|(id, _data)| id)
        }

        pub(super) fn by_name(&self, name: &str) -> Option<RustLibSrcCrate> {
            let (id, _data) = self.crates.iter().find(|(_id, data)| data.name == name)?;
            Some(id)
        }
    }

    pub(crate) type RustLibSrcCrate = Idx<RustLibSrcCrateData>;

    #[derive(Debug, Clone, Eq, PartialEq)]
    pub(crate) struct RustLibSrcCrateData {
        pub(crate) name: String,
        pub(crate) root: ManifestPath,
        pub(crate) deps: Vec<RustLibSrcCrate>,
    }

    pub(super) const SYSROOT_CRATES: &str = "
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

    pub(super) const ALLOC_DEPS: &str = "core";

    pub(super) const STD_DEPS: &str = "
alloc
panic_unwind
panic_abort
core
profiler_builtins
unwind
std_detect
test";

    // core is required for our builtin derives to work in the proc_macro lib currently
    pub(super) const PROC_MACRO_DEPS: &str = "
std
core";
}
