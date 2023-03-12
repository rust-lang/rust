//! Workspace information we get from cargo consists of two pieces. The first is
//! the output of `cargo metadata`. The second is the output of running
//! `build.rs` files (`OUT_DIR` env var, extra cfg flags) and compiling proc
//! macro.
//!
//! This module implements this second part. We use "build script" terminology
//! here, but it covers procedural macros as well.

use std::{
    cell::RefCell,
    io, mem,
    path::{self, PathBuf},
    process::Command,
};

use cargo_metadata::{camino::Utf8Path, Message};
use la_arena::ArenaMap;
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashMap;
use semver::Version;
use serde::Deserialize;

use crate::{
    cfg_flag::CfgFlag, utf8_stdout, CargoConfig, CargoFeatures, CargoWorkspace, InvocationLocation,
    InvocationStrategy, Package,
};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct WorkspaceBuildScripts {
    outputs: ArenaMap<Package, BuildScriptOutput>,
    error: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct BuildScriptOutput {
    /// List of config flags defined by this package's build script.
    pub(crate) cfgs: Vec<CfgFlag>,
    /// List of cargo-related environment variables with their value.
    ///
    /// If the package has a build script which defines environment variables,
    /// they can also be found here.
    pub(crate) envs: Vec<(String, String)>,
    /// Directory where a build script might place its output.
    pub(crate) out_dir: Option<AbsPathBuf>,
    /// Path to the proc-macro library file if this package exposes proc-macros.
    pub(crate) proc_macro_dylib_path: Option<AbsPathBuf>,
}

impl BuildScriptOutput {
    fn is_unchanged(&self) -> bool {
        self.cfgs.is_empty()
            && self.envs.is_empty()
            && self.out_dir.is_none()
            && self.proc_macro_dylib_path.is_none()
    }
}

impl WorkspaceBuildScripts {
    fn build_command(config: &CargoConfig) -> io::Result<Command> {
        let mut cmd = match config.run_build_script_command.as_deref() {
            Some([program, args @ ..]) => {
                let mut cmd = Command::new(program);
                cmd.args(args);
                cmd
            }
            _ => {
                let mut cmd = Command::new(toolchain::cargo());

                cmd.args(["check", "--quiet", "--workspace", "--message-format=json"]);
                cmd.args(&config.extra_args);

                // --all-targets includes tests, benches and examples in addition to the
                // default lib and bins. This is an independent concept from the --target
                // flag below.
                cmd.arg("--all-targets");

                if let Some(target) = &config.target {
                    cmd.args(["--target", target]);
                }

                match &config.features {
                    CargoFeatures::All => {
                        cmd.arg("--all-features");
                    }
                    CargoFeatures::Selected { features, no_default_features } => {
                        if *no_default_features {
                            cmd.arg("--no-default-features");
                        }
                        if !features.is_empty() {
                            cmd.arg("--features");
                            cmd.arg(features.join(" "));
                        }
                    }
                }

                cmd
            }
        };

        cmd.envs(&config.extra_env);
        if config.wrap_rustc_in_build_scripts {
            // Setup RUSTC_WRAPPER to point to `rust-analyzer` binary itself. We use
            // that to compile only proc macros and build scripts during the initial
            // `cargo check`.
            let myself = std::env::current_exe()?;
            cmd.env("RUSTC_WRAPPER", myself);
            cmd.env("RA_RUSTC_WRAPPER", "1");
        }

        Ok(cmd)
    }

    /// Runs the build scripts for the given workspace
    pub(crate) fn run_for_workspace(
        config: &CargoConfig,
        workspace: &CargoWorkspace,
        progress: &dyn Fn(String),
        toolchain: &Option<Version>,
    ) -> io::Result<WorkspaceBuildScripts> {
        const RUST_1_62: Version = Version::new(1, 62, 0);

        let current_dir = match &config.invocation_location {
            InvocationLocation::Root(root) if config.run_build_script_command.is_some() => {
                root.as_path()
            }
            _ => workspace.workspace_root(),
        }
        .as_ref();

        match Self::run_per_ws(Self::build_command(config)?, workspace, current_dir, progress) {
            Ok(WorkspaceBuildScripts { error: Some(error), .. })
                if toolchain.as_ref().map_or(false, |it| *it >= RUST_1_62) =>
            {
                // building build scripts failed, attempt to build with --keep-going so
                // that we potentially get more build data
                let mut cmd = Self::build_command(config)?;
                cmd.args(["-Z", "unstable-options", "--keep-going"]).env("RUSTC_BOOTSTRAP", "1");
                let mut res = Self::run_per_ws(cmd, workspace, current_dir, progress)?;
                res.error = Some(error);
                Ok(res)
            }
            res => res,
        }
    }

    /// Runs the build scripts by invoking the configured command *once*.
    /// This populates the outputs for all passed in workspaces.
    pub(crate) fn run_once(
        config: &CargoConfig,
        workspaces: &[&CargoWorkspace],
        progress: &dyn Fn(String),
    ) -> io::Result<Vec<WorkspaceBuildScripts>> {
        assert_eq!(config.invocation_strategy, InvocationStrategy::Once);

        let current_dir = match &config.invocation_location {
            InvocationLocation::Root(root) => root,
            InvocationLocation::Workspace => {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "Cannot run build scripts from workspace with invocation strategy `once`",
                ))
            }
        };
        let cmd = Self::build_command(config)?;
        // NB: Cargo.toml could have been modified between `cargo metadata` and
        // `cargo check`. We shouldn't assume that package ids we see here are
        // exactly those from `config`.
        let mut by_id = FxHashMap::default();
        // some workspaces might depend on the same crates, so we need to duplicate the outputs
        // to those collisions
        let mut collisions = Vec::new();
        let mut res: Vec<_> = workspaces
            .iter()
            .enumerate()
            .map(|(idx, workspace)| {
                let mut res = WorkspaceBuildScripts::default();
                for package in workspace.packages() {
                    res.outputs.insert(package, BuildScriptOutput::default());
                    if by_id.contains_key(&workspace[package].id) {
                        collisions.push((&workspace[package].id, idx, package));
                    } else {
                        by_id.insert(workspace[package].id.clone(), (package, idx));
                    }
                }
                res
            })
            .collect();

        let errors = Self::run_command(
            cmd,
            current_dir.as_path().as_ref(),
            |package, cb| {
                if let Some(&(package, workspace)) = by_id.get(package) {
                    cb(&workspaces[workspace][package].name, &mut res[workspace].outputs[package]);
                }
            },
            progress,
        )?;
        res.iter_mut().for_each(|it| it.error = errors.clone());
        collisions.into_iter().for_each(|(id, workspace, package)| {
            if let Some(&(p, w)) = by_id.get(id) {
                res[workspace].outputs[package] = res[w].outputs[p].clone();
            }
        });

        if tracing::enabled!(tracing::Level::INFO) {
            for (idx, workspace) in workspaces.iter().enumerate() {
                for package in workspace.packages() {
                    let package_build_data = &mut res[idx].outputs[package];
                    if !package_build_data.is_unchanged() {
                        tracing::info!(
                            "{}: {:?}",
                            workspace[package].manifest.parent().display(),
                            package_build_data,
                        );
                    }
                }
            }
        }

        Ok(res)
    }

    fn run_per_ws(
        cmd: Command,
        workspace: &CargoWorkspace,
        current_dir: &path::Path,
        progress: &dyn Fn(String),
    ) -> io::Result<WorkspaceBuildScripts> {
        let mut res = WorkspaceBuildScripts::default();
        let outputs = &mut res.outputs;
        // NB: Cargo.toml could have been modified between `cargo metadata` and
        // `cargo check`. We shouldn't assume that package ids we see here are
        // exactly those from `config`.
        let mut by_id: FxHashMap<String, Package> = FxHashMap::default();
        for package in workspace.packages() {
            outputs.insert(package, BuildScriptOutput::default());
            by_id.insert(workspace[package].id.clone(), package);
        }

        res.error = Self::run_command(
            cmd,
            current_dir,
            |package, cb| {
                if let Some(&package) = by_id.get(package) {
                    cb(&workspace[package].name, &mut outputs[package]);
                }
            },
            progress,
        )?;

        if tracing::enabled!(tracing::Level::INFO) {
            for package in workspace.packages() {
                let package_build_data = &outputs[package];
                if !package_build_data.is_unchanged() {
                    tracing::info!(
                        "{}: {:?}",
                        workspace[package].manifest.parent().display(),
                        package_build_data,
                    );
                }
            }
        }

        Ok(res)
    }

    fn run_command(
        mut cmd: Command,
        current_dir: &path::Path,
        // ideally this would be something like:
        // with_output_for: impl FnMut(&str, dyn FnOnce(&mut BuildScriptOutput)),
        // but owned trait objects aren't a thing
        mut with_output_for: impl FnMut(&str, &mut dyn FnMut(&str, &mut BuildScriptOutput)),
        progress: &dyn Fn(String),
    ) -> io::Result<Option<String>> {
        let errors = RefCell::new(String::new());
        let push_err = |err: &str| {
            let mut e = errors.borrow_mut();
            e.push_str(err);
            e.push('\n');
        };

        tracing::info!("Running build scripts in {}: {:?}", current_dir.display(), cmd);
        cmd.current_dir(current_dir);
        let output = stdx::process::spawn_with_streaming_output(
            cmd,
            &mut |line| {
                // Copy-pasted from existing cargo_metadata. It seems like we
                // should be using serde_stacker here?
                let mut deserializer = serde_json::Deserializer::from_str(line);
                deserializer.disable_recursion_limit();
                let message = Message::deserialize(&mut deserializer)
                    .unwrap_or_else(|_| Message::TextLine(line.to_string()));

                match message {
                    Message::BuildScriptExecuted(mut message) => {
                        with_output_for(&message.package_id.repr, &mut |name, data| {
                            progress(format!("running build-script: {name}"));
                            let cfgs = {
                                let mut acc = Vec::new();
                                for cfg in &message.cfgs {
                                    match cfg.parse::<CfgFlag>() {
                                        Ok(it) => acc.push(it),
                                        Err(err) => {
                                            push_err(&format!(
                                                "invalid cfg from cargo-metadata: {err}"
                                            ));
                                            return;
                                        }
                                    };
                                }
                                acc
                            };
                            if !message.env.is_empty() {
                                data.envs = mem::take(&mut message.env);
                            }
                            // cargo_metadata crate returns default (empty) path for
                            // older cargos, which is not absolute, so work around that.
                            let out_dir = mem::take(&mut message.out_dir).into_os_string();
                            if !out_dir.is_empty() {
                                let out_dir = AbsPathBuf::assert(PathBuf::from(out_dir));
                                // inject_cargo_env(package, package_build_data);
                                // NOTE: cargo and rustc seem to hide non-UTF-8 strings from env! and option_env!()
                                if let Some(out_dir) =
                                    out_dir.as_os_str().to_str().map(|s| s.to_owned())
                                {
                                    data.envs.push(("OUT_DIR".to_string(), out_dir));
                                }
                                data.out_dir = Some(out_dir);
                                data.cfgs = cfgs;
                            }
                        });
                    }
                    Message::CompilerArtifact(message) => {
                        with_output_for(&message.package_id.repr, &mut |name, data| {
                            progress(format!("building proc-macros: {name}"));
                            if message.target.kind.iter().any(|k| k == "proc-macro") {
                                // Skip rmeta file
                                if let Some(filename) =
                                    message.filenames.iter().find(|name| is_dylib(name))
                                {
                                    let filename = AbsPathBuf::assert(PathBuf::from(&filename));
                                    data.proc_macro_dylib_path = Some(filename);
                                }
                            }
                        });
                    }
                    Message::CompilerMessage(message) => {
                        progress(message.target.name);

                        if let Some(diag) = message.message.rendered.as_deref() {
                            push_err(diag);
                        }
                    }
                    Message::BuildFinished(_) => {}
                    Message::TextLine(_) => {}
                    _ => {}
                }
            },
            &mut |line| {
                push_err(line);
            },
        )?;

        let errors = if !output.status.success() {
            let errors = errors.into_inner();
            Some(if errors.is_empty() { "cargo check failed".to_string() } else { errors })
        } else {
            None
        };
        Ok(errors)
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub(crate) fn get_output(&self, idx: Package) -> Option<&BuildScriptOutput> {
        self.outputs.get(idx)
    }

    pub(crate) fn rustc_crates(
        rustc: &CargoWorkspace,
        current_dir: &AbsPath,
        extra_env: &FxHashMap<String, String>,
    ) -> Self {
        let mut bs = WorkspaceBuildScripts::default();
        for p in rustc.packages() {
            bs.outputs.insert(p, BuildScriptOutput::default());
        }
        let res = (|| {
            let target_libdir = (|| {
                let mut cargo_config = Command::new(toolchain::cargo());
                cargo_config.envs(extra_env);
                cargo_config
                    .current_dir(current_dir)
                    .args(["rustc", "-Z", "unstable-options", "--print", "target-libdir"])
                    .env("RUSTC_BOOTSTRAP", "1");
                if let Ok(it) = utf8_stdout(cargo_config) {
                    return Ok(it);
                }
                let mut cmd = Command::new(toolchain::rustc());
                cmd.envs(extra_env);
                cmd.args(["--print", "target-libdir"]);
                utf8_stdout(cmd)
            })()?;

            let target_libdir = AbsPathBuf::try_from(PathBuf::from(target_libdir))
                .map_err(|_| anyhow::format_err!("target-libdir was not an absolute path"))?;
            tracing::info!("Loading rustc proc-macro paths from {}", target_libdir.display());

            let proc_macro_dylibs: Vec<(String, AbsPathBuf)> = std::fs::read_dir(target_libdir)?
                .filter_map(|entry| {
                    let dir_entry = entry.ok()?;
                    if dir_entry.file_type().ok()?.is_file() {
                        let path = dir_entry.path();
                        tracing::info!("p{:?}", path);
                        let extension = path.extension()?;
                        if extension == std::env::consts::DLL_EXTENSION {
                            let name = path.file_stem()?.to_str()?.split_once('-')?.0.to_owned();
                            let path = AbsPathBuf::try_from(path).ok()?;
                            return Some((name, path));
                        }
                    }
                    None
                })
                .collect();
            for p in rustc.packages() {
                let package = &rustc[p];
                if package.targets.iter().any(|&it| rustc[it].is_proc_macro) {
                    if let Some((_, path)) =
                        proc_macro_dylibs.iter().find(|(name, _)| *name == package.name)
                    {
                        bs.outputs[p].proc_macro_dylib_path = Some(path.clone());
                    }
                }
            }

            if tracing::enabled!(tracing::Level::INFO) {
                for package in rustc.packages() {
                    let package_build_data = &bs.outputs[package];
                    if !package_build_data.is_unchanged() {
                        tracing::info!(
                            "{}: {:?}",
                            rustc[package].manifest.parent().display(),
                            package_build_data,
                        );
                    }
                }
            }
            Ok(())
        })();
        if let Err::<_, anyhow::Error>(e) = res {
            bs.error = Some(e.to_string());
        }
        bs
    }
}

// FIXME: Find a better way to know if it is a dylib.
fn is_dylib(path: &Utf8Path) -> bool {
    match path.extension().map(|e| e.to_string().to_lowercase()) {
        None => false,
        Some(ext) => matches!(ext.as_str(), "dll" | "dylib" | "so"),
    }
}
