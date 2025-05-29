//! This module contains the core logic for parsing, validating, and inferring the
//! final bootstrap `Config` from various raw inputs.
//!
//! It handles the intricate process of reading command-line arguments, environment variables,
//! and `bootstrap.toml` files, merging them, applying defaults, and performing
//! cross-component validations. The main `parse_inner` function and its related
//! helper functions reside here, transforming the raw `Toml` data into the structured `Config` types

use std::collections::HashMap;
use std::path::{Path, PathBuf, absolute};
use std::{cmp, env, fs};

use build_helper::ci::CiEnv;
use build_helper::exit;
use serde::Deserialize;
#[cfg(feature = "tracing")]
use tracing::{instrument, span};

use crate::core::config::flags::Flags;
pub use crate::core::config::flags::Subcommand;
use crate::core::config::target_selection::TargetSelectionList;
use crate::core::config::toml::TomlConfig;
use crate::core::config::toml::build::Build;
use crate::core::config::toml::common::{ReplaceOpt, StringOrBool};
use crate::core::config::toml::merge::Merge;
use crate::core::config::toml::rust::LldMode;
use crate::core::config::toml::target::Target;
use crate::core::config::{set, threads_from_config};
use crate::utils::channel::GitInfo;
use crate::utils::helpers::{self, exe, t};
use crate::{Config, DryRun, TargetSelection, check_ci_llvm};

impl Config {
    #[cfg_attr(
        feature = "tracing",
        instrument(
            target = "CONFIG_HANDLING",
            level = "trace",
            name = "Config::parse_inner",
            skip_all
        )
    )]
    pub(crate) fn parse_inner(
        mut flags: Flags,
        get_toml: impl Fn(&Path) -> Result<TomlConfig, toml::de::Error>,
    ) -> Config {
        let mut config = Config::default_opts();

        // Set flags.
        config.paths = std::mem::take(&mut flags.paths);

        #[cfg(feature = "tracing")]
        span!(
            target: "CONFIG_HANDLING",
            tracing::Level::TRACE,
            "collecting paths and path exclusions",
            "flags.paths" = ?flags.paths,
            "flags.skip" = ?flags.skip,
            "flags.exclude" = ?flags.exclude
        );

        #[cfg(feature = "tracing")]
        span!(
            target: "CONFIG_HANDLING",
            tracing::Level::TRACE,
            "normalizing and combining `flag.skip`/`flag.exclude` paths",
            "config.skip" = ?config.skip,
        );

        config.include_default_paths = flags.include_default_paths;
        config.rustc_error_format = flags.rustc_error_format;
        config.json_output = flags.json_output;
        config.on_fail = flags.on_fail;
        config.cmd = flags.cmd;
        config.incremental = flags.incremental;
        config.dry_run = if flags.dry_run { DryRun::UserSelected } else { DryRun::Disabled };
        config.dump_bootstrap_shims = flags.dump_bootstrap_shims;
        config.keep_stage = flags.keep_stage;
        config.keep_stage_std = flags.keep_stage_std;
        config.color = flags.color;
        config.free_args = std::mem::take(&mut flags.free_args);
        config.llvm_profile_use = flags.llvm_profile_use;
        config.llvm_profile_generate = flags.llvm_profile_generate;
        config.enable_bolt_settings = flags.enable_bolt_settings;
        config.bypass_bootstrap_lock = flags.bypass_bootstrap_lock;
        config.is_running_on_ci = flags.ci.unwrap_or(CiEnv::is_ci());

        // Infer the rest of the configuration.

        if let Some(src) = flags.src {
            config.src = src
        } else {
            // Infer the source directory. This is non-trivial because we want to support a downloaded bootstrap binary,
            // running on a completely different machine from where it was compiled.
            let mut cmd = helpers::git(None);
            // NOTE: we cannot support running from outside the repository because the only other path we have available
            // is set at compile time, which can be wrong if bootstrap was downloaded rather than compiled locally.
            // We still support running outside the repository if we find we aren't in a git directory.

            // NOTE: We get a relative path from git to work around an issue on MSYS/mingw. If we used an absolute path,
            // and end up using MSYS's git rather than git-for-windows, we would get a unix-y MSYS path. But as bootstrap
            // has already been (kinda-cross-)compiled to Windows land, we require a normal Windows path.
            cmd.arg("rev-parse").arg("--show-cdup");
            // Discard stderr because we expect this to fail when building from a tarball.
            let output = cmd
                .as_command_mut()
                .stderr(std::process::Stdio::null())
                .output()
                .ok()
                .and_then(|output| if output.status.success() { Some(output) } else { None });
            if let Some(output) = output {
                let git_root_relative = String::from_utf8(output.stdout).unwrap();
                // We need to canonicalize this path to make sure it uses backslashes instead of forward slashes,
                // and to resolve any relative components.
                let git_root = env::current_dir()
                    .unwrap()
                    .join(PathBuf::from(git_root_relative.trim()))
                    .canonicalize()
                    .unwrap();
                let s = git_root.to_str().unwrap();

                // Bootstrap is quite bad at handling /? in front of paths
                let git_root = match s.strip_prefix("\\\\?\\") {
                    Some(p) => PathBuf::from(p),
                    None => git_root,
                };
                // If this doesn't have at least `stage0`, we guessed wrong. This can happen when,
                // for example, the build directory is inside of another unrelated git directory.
                // In that case keep the original `CARGO_MANIFEST_DIR` handling.
                //
                // NOTE: this implies that downloadable bootstrap isn't supported when the build directory is outside
                // the source directory. We could fix that by setting a variable from all three of python, ./x, and x.ps1.
                if git_root.join("src").join("stage0").exists() {
                    config.src = git_root;
                }
            } else {
                // We're building from a tarball, not git sources.
                // We don't support pre-downloaded bootstrap in this case.
            }
        }

        if cfg!(test) {
            // Use the build directory of the original x.py invocation, so that we can set `initial_rustc` properly.
            config.out = Path::new(
                &env::var_os("CARGO_TARGET_DIR").expect("cargo test directly is not supported"),
            )
            .parent()
            .unwrap()
            .to_path_buf();
        }

        config.stage0_metadata = build_helper::stage0_parser::parse_stage0_file();

        // Locate the configuration file using the following priority (first match wins):
        // 1. `--config <path>` (explicit flag)
        // 2. `RUST_BOOTSTRAP_CONFIG` environment variable
        // 3. `./bootstrap.toml` (local file)
        // 4. `<root>/bootstrap.toml`
        // 5. `./config.toml` (fallback for backward compatibility)
        // 6. `<root>/config.toml`
        let toml_path = flags
            .config
            .clone()
            .or_else(|| env::var_os("RUST_BOOTSTRAP_CONFIG").map(PathBuf::from));
        let using_default_path = toml_path.is_none();
        let mut toml_path = toml_path.unwrap_or_else(|| PathBuf::from("bootstrap.toml"));

        if using_default_path && !toml_path.exists() {
            toml_path = config.src.join(PathBuf::from("bootstrap.toml"));
            if !toml_path.exists() {
                toml_path = PathBuf::from("config.toml");
                if !toml_path.exists() {
                    toml_path = config.src.join(PathBuf::from("config.toml"));
                }
            }
        }

        // Give a hard error if `--config` or `RUST_BOOTSTRAP_CONFIG` are set to a missing path,
        // but not if `bootstrap.toml` hasn't been created.
        let mut toml = if !using_default_path || toml_path.exists() {
            config.config = Some(if cfg!(not(test)) {
                toml_path = toml_path.canonicalize().unwrap();
                toml_path.clone()
            } else {
                toml_path.clone()
            });
            get_toml(&toml_path).unwrap_or_else(|e| {
                eprintln!("ERROR: Failed to parse '{}': {e}", toml_path.display());
                exit!(2);
            })
        } else {
            config.config = None;
            TomlConfig::default()
        };

        if cfg!(test) {
            // When configuring bootstrap for tests, make sure to set the rustc and Cargo to the
            // same ones used to call the tests (if custom ones are not defined in the toml). If we
            // don't do that, bootstrap will use its own detection logic to find a suitable rustc
            // and Cargo, which doesn't work when the caller is specìfying a custom local rustc or
            // Cargo in their bootstrap.toml.
            let build = toml.build.get_or_insert_with(Default::default);
            build.rustc = build.rustc.take().or(std::env::var_os("RUSTC").map(|p| p.into()));
            build.cargo = build.cargo.take().or(std::env::var_os("CARGO").map(|p| p.into()));
        }

        if GitInfo::new(false, &config.src).is_from_tarball() && toml.profile.is_none() {
            toml.profile = Some("dist".into());
        }

        // Reverse the list to ensure the last added config extension remains the most dominant.
        // For example, given ["a.toml", "b.toml"], "b.toml" should take precedence over "a.toml".
        //
        // This must be handled before applying the `profile` since `include`s should always take
        // precedence over `profile`s.
        for include_path in toml.include.clone().unwrap_or_default().iter().rev() {
            let include_path = toml_path.parent().unwrap().join(include_path);

            let included_toml = get_toml(&include_path).unwrap_or_else(|e| {
                eprintln!("ERROR: Failed to parse '{}': {e}", include_path.display());
                exit!(2);
            });
            toml.merge(
                Some(include_path),
                &mut Default::default(),
                included_toml,
                ReplaceOpt::IgnoreDuplicate,
            );
        }

        if let Some(include) = &toml.profile {
            // Allows creating alias for profile names, allowing
            // profiles to be renamed while maintaining back compatibility
            // Keep in sync with `profile_aliases` in bootstrap.py
            let profile_aliases = HashMap::from([("user", "dist")]);
            let include = match profile_aliases.get(include.as_str()) {
                Some(alias) => alias,
                None => include.as_str(),
            };
            let mut include_path = config.src.clone();
            include_path.push("src");
            include_path.push("bootstrap");
            include_path.push("defaults");
            include_path.push(format!("bootstrap.{include}.toml"));
            let included_toml = get_toml(&include_path).unwrap_or_else(|e| {
                eprintln!(
                    "ERROR: Failed to parse default config profile at '{}': {e}",
                    include_path.display()
                );
                exit!(2);
            });
            toml.merge(
                Some(include_path),
                &mut Default::default(),
                included_toml,
                ReplaceOpt::IgnoreDuplicate,
            );
        }

        let mut override_toml = TomlConfig::default();
        for option in flags.set.iter() {
            fn get_table(option: &str) -> Result<TomlConfig, toml::de::Error> {
                toml::from_str(option).and_then(|table: toml::Value| TomlConfig::deserialize(table))
            }

            let mut err = match get_table(option) {
                Ok(v) => {
                    override_toml.merge(
                        None,
                        &mut Default::default(),
                        v,
                        ReplaceOpt::ErrorOnDuplicate,
                    );
                    continue;
                }
                Err(e) => e,
            };
            // We want to be able to set string values without quotes,
            // like in `configure.py`. Try adding quotes around the right hand side
            if let Some((key, value)) = option.split_once('=') {
                if !value.contains('"') {
                    match get_table(&format!(r#"{key}="{value}""#)) {
                        Ok(v) => {
                            override_toml.merge(
                                None,
                                &mut Default::default(),
                                v,
                                ReplaceOpt::ErrorOnDuplicate,
                            );
                            continue;
                        }
                        Err(e) => err = e,
                    }
                }
            }
            eprintln!("failed to parse override `{option}`: `{err}");
            exit!(2)
        }
        toml.merge(None, &mut Default::default(), override_toml, ReplaceOpt::Override);

        config.change_id = toml.change_id.inner;

        let Build {
            mut description,
            build,
            host,
            target,
            build_dir,
            cargo,
            rustc,
            rustfmt,
            cargo_clippy,
            docs,
            compiler_docs,
            library_docs_private_items,
            docs_minification,
            submodules,
            gdb,
            lldb,
            nodejs,
            npm,
            python,
            reuse,
            locked_deps,
            vendor,
            full_bootstrap,
            bootstrap_cache_path,
            extended,
            tools,
            verbose,
            sanitizers,
            profiler,
            cargo_native_static,
            low_priority,
            configure_args,
            local_rebuild,
            print_step_timings,
            print_step_rusage,
            check_stage,
            doc_stage,
            build_stage,
            test_stage,
            install_stage,
            dist_stage,
            bench_stage,
            patch_binaries_for_nix,
            // This field is only used by bootstrap.py
            metrics: _,
            android_ndk,
            optimized_compiler_builtins,
            jobs,
            compiletest_diff_tool,
            compiletest_use_stage0_libtest,
            mut ccache,
            exclude,
        } = toml.build.unwrap_or_default();

        let mut paths: Vec<PathBuf> = flags.skip.into_iter().chain(flags.exclude).collect();

        if let Some(exclude) = exclude {
            paths.extend(exclude);
        }

        config.skip = paths
            .into_iter()
            .map(|p| {
                // Never return top-level path here as it would break `--skip`
                // logic on rustc's internal test framework which is utilized
                // by compiletest.
                if cfg!(windows) {
                    PathBuf::from(p.to_str().unwrap().replace('/', "\\"))
                } else {
                    p
                }
            })
            .collect();

        config.jobs = Some(threads_from_config(flags.jobs.unwrap_or(jobs.unwrap_or(0))));

        if let Some(file_build) = build {
            config.build = TargetSelection::from_user(&file_build);
        };

        set(&mut config.out, flags.build_dir.or_else(|| build_dir.map(PathBuf::from)));
        // NOTE: Bootstrap spawns various commands with different working directories.
        // To avoid writing to random places on the file system, `config.out` needs to be an absolute path.
        if !config.out.is_absolute() {
            // `canonicalize` requires the path to already exist. Use our vendored copy of `absolute` instead.
            config.out = absolute(&config.out).expect("can't make empty path absolute");
        }

        if cargo_clippy.is_some() && rustc.is_none() {
            println!(
                "WARNING: Using `build.cargo-clippy` without `build.rustc` usually fails due to toolchain conflict."
            );
        }

        config.initial_rustc = if let Some(rustc) = rustc {
            if !flags.skip_stage0_validation {
                config.check_stage0_version(&rustc, "rustc");
            }
            rustc
        } else {
            config.download_beta_toolchain();
            config
                .out
                .join(config.build)
                .join("stage0")
                .join("bin")
                .join(exe("rustc", config.build))
        };

        config.initial_sysroot = config.initial_rustc.ancestors().nth(2).unwrap().into();

        config.initial_cargo_clippy = cargo_clippy;

        config.initial_cargo = if let Some(cargo) = cargo {
            if !flags.skip_stage0_validation {
                config.check_stage0_version(&cargo, "cargo");
            }
            cargo
        } else {
            config.download_beta_toolchain();
            config.initial_sysroot.join("bin").join(exe("cargo", config.build))
        };

        // NOTE: it's important this comes *after* we set `initial_rustc` just above.
        if config.dry_run() {
            let dir = config.out.join("tmp-dry-run");
            t!(fs::create_dir_all(&dir));
            config.out = dir;
        }

        config.hosts = if let Some(TargetSelectionList(arg_host)) = flags.host {
            arg_host
        } else if let Some(file_host) = host {
            file_host.iter().map(|h| TargetSelection::from_user(h)).collect()
        } else {
            vec![config.build]
        };
        config.targets = if let Some(TargetSelectionList(arg_target)) = flags.target {
            arg_target
        } else if let Some(file_target) = target {
            file_target.iter().map(|h| TargetSelection::from_user(h)).collect()
        } else {
            // If target is *not* configured, then default to the host
            // toolchains.
            config.hosts.clone()
        };

        config.nodejs = nodejs.map(PathBuf::from);
        config.npm = npm.map(PathBuf::from);
        config.gdb = gdb.map(PathBuf::from);
        config.lldb = lldb.map(PathBuf::from);
        config.python = python.map(PathBuf::from);
        config.reuse = reuse.map(PathBuf::from);
        config.submodules = submodules;
        config.android_ndk = android_ndk;
        config.bootstrap_cache_path = bootstrap_cache_path;
        set(&mut config.low_priority, low_priority);
        set(&mut config.compiler_docs, compiler_docs);
        set(&mut config.library_docs_private_items, library_docs_private_items);
        set(&mut config.docs_minification, docs_minification);
        set(&mut config.docs, docs);
        set(&mut config.locked_deps, locked_deps);
        set(&mut config.full_bootstrap, full_bootstrap);
        set(&mut config.extended, extended);
        config.tools = tools;
        set(&mut config.verbose, verbose);
        set(&mut config.sanitizers, sanitizers);
        set(&mut config.profiler, profiler);
        set(&mut config.cargo_native_static, cargo_native_static);
        set(&mut config.configure_args, configure_args);
        set(&mut config.local_rebuild, local_rebuild);
        set(&mut config.print_step_timings, print_step_timings);
        set(&mut config.print_step_rusage, print_step_rusage);
        config.patch_binaries_for_nix = patch_binaries_for_nix;

        config.verbose = cmp::max(config.verbose, flags.verbose as usize);

        // Verbose flag is a good default for `rust.verbose-tests`.
        config.verbose_tests = config.is_verbose();

        config.apply_install_config(toml.install);

        config.llvm_assertions =
            toml.llvm.as_ref().is_some_and(|llvm| llvm.assertions.unwrap_or(false));

        let file_content = t!(fs::read_to_string(config.src.join("src/ci/channel")));
        let ci_channel = file_content.trim_end();

        let toml_channel = toml.rust.as_ref().and_then(|r| r.channel.clone());
        let is_user_configured_rust_channel = match toml_channel {
            Some(channel) if channel == "auto-detect" => {
                config.channel = ci_channel.into();
                true
            }
            Some(channel) => {
                config.channel = channel;
                true
            }
            None => false,
        };

        let default = config.channel == "dev";
        config.omit_git_hash = toml.rust.as_ref().and_then(|r| r.omit_git_hash).unwrap_or(default);

        config.rust_info = GitInfo::new(config.omit_git_hash, &config.src);
        config.cargo_info = GitInfo::new(config.omit_git_hash, &config.src.join("src/tools/cargo"));
        config.rust_analyzer_info =
            GitInfo::new(config.omit_git_hash, &config.src.join("src/tools/rust-analyzer"));
        config.clippy_info =
            GitInfo::new(config.omit_git_hash, &config.src.join("src/tools/clippy"));
        config.miri_info = GitInfo::new(config.omit_git_hash, &config.src.join("src/tools/miri"));
        config.rustfmt_info =
            GitInfo::new(config.omit_git_hash, &config.src.join("src/tools/rustfmt"));
        config.enzyme_info =
            GitInfo::new(config.omit_git_hash, &config.src.join("src/tools/enzyme"));
        config.in_tree_llvm_info = GitInfo::new(false, &config.src.join("src/llvm-project"));
        config.in_tree_gcc_info = GitInfo::new(false, &config.src.join("src/gcc"));

        config.vendor = vendor.unwrap_or(
            config.rust_info.is_from_tarball()
                && config.src.join("vendor").exists()
                && config.src.join(".cargo/config.toml").exists(),
        );

        if !is_user_configured_rust_channel && config.rust_info.is_from_tarball() {
            config.channel = ci_channel.into();
        }

        config.rust_profile_use = flags.rust_profile_use;
        config.rust_profile_generate = flags.rust_profile_generate;

        config.apply_rust_config(toml.rust, flags.warnings, &mut description);

        config.reproducible_artifacts = flags.reproducible_artifact;
        config.description = description;

        // We need to override `rust.channel` if it's manually specified when using the CI rustc.
        // This is because if the compiler uses a different channel than the one specified in bootstrap.toml,
        // tests may fail due to using a different channel than the one used by the compiler during tests.
        if let Some(commit) = &config.download_rustc_commit {
            if is_user_configured_rust_channel {
                println!(
                    "WARNING: `rust.download-rustc` is enabled. The `rust.channel` option will be overridden by the CI rustc's channel."
                );

                let channel = config
                    .read_file_by_commit(Path::new("src/ci/channel"), commit)
                    .trim()
                    .to_owned();

                config.channel = channel;
            }
        }

        config.apply_llvm_config(toml.llvm, &mut ccache);

        config.apply_gcc_config(toml.gcc);

        config.apply_target_config(toml.target);

        match ccache {
            Some(StringOrBool::String(ref s)) => config.ccache = Some(s.to_string()),
            Some(StringOrBool::Bool(true)) => {
                config.ccache = Some("ccache".to_string());
            }
            Some(StringOrBool::Bool(false)) | None => {}
        }

        if config.llvm_from_ci {
            let triple = &config.build.triple;
            let ci_llvm_bin = config.ci_llvm_root().join("bin");
            let build_target = config
                .target_config
                .entry(config.build)
                .or_insert_with(|| Target::from_triple(triple));

            check_ci_llvm!(build_target.llvm_config);
            check_ci_llvm!(build_target.llvm_filecheck);
            build_target.llvm_config = Some(ci_llvm_bin.join(exe("llvm-config", config.build)));
            build_target.llvm_filecheck = Some(ci_llvm_bin.join(exe("FileCheck", config.build)));
        }

        config.apply_dist_config(toml.dist);

        config.initial_rustfmt =
            if let Some(r) = rustfmt { Some(r) } else { config.maybe_download_rustfmt() };

        if matches!(config.lld_mode, LldMode::SelfContained)
            && !config.lld_enabled
            && flags.stage.unwrap_or(0) > 0
        {
            panic!(
                "Trying to use self-contained lld as a linker, but LLD is not being added to the sysroot. Enable it with rust.lld = true."
            );
        }

        if config.lld_enabled && config.is_system_llvm(config.build) {
            eprintln!(
                "Warning: LLD is enabled when using external llvm-config. LLD will not be built and copied to the sysroot."
            );
        }

        config.optimized_compiler_builtins =
            optimized_compiler_builtins.unwrap_or(config.channel != "dev");
        config.compiletest_diff_tool = compiletest_diff_tool;
        config.compiletest_use_stage0_libtest = compiletest_use_stage0_libtest.unwrap_or(true);

        let download_rustc = config.download_rustc_commit.is_some();
        config.explicit_stage_from_cli = flags.stage.is_some();
        config.explicit_stage_from_config = test_stage.is_some()
            || build_stage.is_some()
            || doc_stage.is_some()
            || dist_stage.is_some()
            || install_stage.is_some()
            || check_stage.is_some()
            || bench_stage.is_some();
        // See https://github.com/rust-lang/compiler-team/issues/326
        config.stage = match config.cmd {
            Subcommand::Check { .. } => flags.stage.or(check_stage).unwrap_or(0),
            // `download-rustc` only has a speed-up for stage2 builds. Default to stage2 unless explicitly overridden.
            Subcommand::Doc { .. } => {
                flags.stage.or(doc_stage).unwrap_or(if download_rustc { 2 } else { 0 })
            }
            Subcommand::Build => {
                flags.stage.or(build_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Test { .. } | Subcommand::Miri { .. } => {
                flags.stage.or(test_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Bench { .. } => flags.stage.or(bench_stage).unwrap_or(2),
            Subcommand::Dist => flags.stage.or(dist_stage).unwrap_or(2),
            Subcommand::Install => flags.stage.or(install_stage).unwrap_or(2),
            Subcommand::Perf { .. } => flags.stage.unwrap_or(1),
            // These are all bootstrap tools, which don't depend on the compiler.
            // The stage we pass shouldn't matter, but use 0 just in case.
            Subcommand::Clean { .. }
            | Subcommand::Clippy { .. }
            | Subcommand::Fix
            | Subcommand::Run { .. }
            | Subcommand::Setup { .. }
            | Subcommand::Format { .. }
            | Subcommand::Suggest { .. }
            | Subcommand::Vendor { .. } => flags.stage.unwrap_or(0),
        };

        // CI should always run stage 2 builds, unless it specifically states otherwise
        #[cfg(not(test))]
        if flags.stage.is_none() && config.is_running_on_ci {
            match config.cmd {
                Subcommand::Test { .. }
                | Subcommand::Miri { .. }
                | Subcommand::Doc { .. }
                | Subcommand::Build
                | Subcommand::Bench { .. }
                | Subcommand::Dist
                | Subcommand::Install => {
                    assert_eq!(
                        config.stage, 2,
                        "x.py should be run with `--stage 2` on CI, but was run with `--stage {}`",
                        config.stage,
                    );
                }
                Subcommand::Clean { .. }
                | Subcommand::Check { .. }
                | Subcommand::Clippy { .. }
                | Subcommand::Fix
                | Subcommand::Run { .. }
                | Subcommand::Setup { .. }
                | Subcommand::Format { .. }
                | Subcommand::Suggest { .. }
                | Subcommand::Vendor { .. }
                | Subcommand::Perf { .. } => {}
            }
        }

        config
    }
}
