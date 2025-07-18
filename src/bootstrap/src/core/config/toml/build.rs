//! This module defines the `Build` struct, which represents the `[build]` table
//! in the `bootstrap.toml` configuration file.
//!
//! The `[build]` table contains global options that influence the overall build process,
//! such as default host and target triples, paths to tools, build directories, and
//! various feature flags. These options apply across different stages and components
//! unless specifically overridden by other configuration sections or command-line flags.

use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Deserializer};

use crate::core::config::target_selection::TargetSelectionList;
use crate::core::config::toml::ReplaceOpt;
use crate::core::config::{Merge, StringOrBool, TargetSelection, set, threads_from_config};
use crate::helpers::exe;
use crate::{Config, HashSet, PathBuf, Subcommand, command, define_config, exit, t};

define_config! {
    /// TOML representation of various global build decisions.
    #[derive(Default)]
    struct Build {
        build: Option<String> = "build",
        description: Option<String> = "description",
        host: Option<Vec<String>> = "host",
        target: Option<Vec<String>> = "target",
        build_dir: Option<String> = "build-dir",
        cargo: Option<PathBuf> = "cargo",
        rustc: Option<PathBuf> = "rustc",
        rustfmt: Option<PathBuf> = "rustfmt",
        cargo_clippy: Option<PathBuf> = "cargo-clippy",
        docs: Option<bool> = "docs",
        compiler_docs: Option<bool> = "compiler-docs",
        library_docs_private_items: Option<bool> = "library-docs-private-items",
        docs_minification: Option<bool> = "docs-minification",
        submodules: Option<bool> = "submodules",
        gdb: Option<String> = "gdb",
        lldb: Option<String> = "lldb",
        nodejs: Option<String> = "nodejs",
        npm: Option<String> = "npm",
        python: Option<String> = "python",
        reuse: Option<String> = "reuse",
        locked_deps: Option<bool> = "locked-deps",
        vendor: Option<bool> = "vendor",
        full_bootstrap: Option<bool> = "full-bootstrap",
        bootstrap_cache_path: Option<PathBuf> = "bootstrap-cache-path",
        extended: Option<bool> = "extended",
        tools: Option<HashSet<String>> = "tools",
        tool: Option<HashMap<String, Tool>> = "tool",
        verbose: Option<usize> = "verbose",
        sanitizers: Option<bool> = "sanitizers",
        profiler: Option<bool> = "profiler",
        cargo_native_static: Option<bool> = "cargo-native-static",
        low_priority: Option<bool> = "low-priority",
        configure_args: Option<Vec<String>> = "configure-args",
        local_rebuild: Option<bool> = "local-rebuild",
        print_step_timings: Option<bool> = "print-step-timings",
        print_step_rusage: Option<bool> = "print-step-rusage",
        check_stage: Option<u32> = "check-stage",
        doc_stage: Option<u32> = "doc-stage",
        build_stage: Option<u32> = "build-stage",
        test_stage: Option<u32> = "test-stage",
        install_stage: Option<u32> = "install-stage",
        dist_stage: Option<u32> = "dist-stage",
        bench_stage: Option<u32> = "bench-stage",
        patch_binaries_for_nix: Option<bool> = "patch-binaries-for-nix",
        // NOTE: only parsed by bootstrap.py, `--feature build-metrics` enables metrics unconditionally
        metrics: Option<bool> = "metrics",
        android_ndk: Option<PathBuf> = "android-ndk",
        optimized_compiler_builtins: Option<bool> = "optimized-compiler-builtins",
        jobs: Option<u32> = "jobs",
        compiletest_diff_tool: Option<String> = "compiletest-diff-tool",
        compiletest_use_stage0_libtest: Option<bool> = "compiletest-use-stage0-libtest",
        tidy_extra_checks: Option<String> = "tidy-extra-checks",
        ccache: Option<StringOrBool> = "ccache",
        exclude: Option<Vec<PathBuf>> = "exclude",
    }
}

define_config! {
    /// Configuration specific for some tool, e.g. which features to enable during build.
    #[derive(Default, Clone)]
    struct Tool {
        features: Option<Vec<String>> = "features",
    }
}

impl Config {
    pub fn apply_build_config(
        &mut self,
        toml_build: Option<Build>,
        flags_skip_stage0_validation: bool,
        flags_stage: Option<u32>,
        flags_host: Option<TargetSelectionList>,
        flags_target: Option<TargetSelectionList>,
        flags_build_dir: Option<PathBuf>,
    ) {
        let Build {
            description,
            build: _,
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
            tool,
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
            tidy_extra_checks,
            ccache,
            exclude,
        } = toml_build.unwrap_or_default();

        if let Some(exclude) = exclude {
            self.skip.extend(
                exclude
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
                    .collect::<Vec<PathBuf>>(),
            );
        }

        self.description = description;
        match ccache {
            Some(StringOrBool::String(ref s)) => self.ccache = Some(s.to_string()),
            Some(StringOrBool::Bool(true)) => {
                self.ccache = Some("ccache".to_string());
            }
            Some(StringOrBool::Bool(false)) | None => {}
        }

        if self.jobs.is_none() {
            self.jobs = Some(threads_from_config(jobs.unwrap_or(0)));
        }

        set(&mut self.out, flags_build_dir.or_else(|| build_dir.map(PathBuf::from)));

        // NOTE: Bootstrap spawns various commands with different working directories.
        // To avoid writing to random places on the file system, `config.out` needs to be an absolute path.
        if !self.out.is_absolute() {
            // `canonicalize` requires the path to already exist. Use our vendored copy of `absolute` instead.
            self.out = std::path::absolute(&self.out).expect("can't make empty path absolute");
        }

        if cargo_clippy.is_some() && rustc.is_none() {
            println!(
                "WARNING: Using `build.cargo-clippy` without `build.rustc` usually fails due to toolchain conflict."
            );
        }

        self.initial_rustc = if let Some(rustc) = rustc {
            if !flags_skip_stage0_validation {
                self.check_stage0_version(&rustc, "rustc");
            }
            rustc
        } else {
            self.download_beta_toolchain();
            self.out
                .join(self.host_target)
                .join("stage0")
                .join("bin")
                .join(exe("rustc", self.host_target))
        };

        self.initial_sysroot = t!(PathBuf::from_str(
            command(&self.initial_rustc)
                .args(["--print", "sysroot"])
                .run_in_dry_run()
                .run_capture_stdout(&self)
                .stdout()
                .trim()
        ));

        self.initial_cargo_clippy = cargo_clippy;

        self.initial_cargo = if let Some(cargo) = cargo {
            if !flags_skip_stage0_validation {
                self.check_stage0_version(&cargo, "cargo");
            }
            cargo
        } else {
            self.download_beta_toolchain();
            self.initial_sysroot.join("bin").join(exe("cargo", self.host_target))
        };

        // NOTE: it's important this comes *after* we set `initial_rustc` just above.
        if self.dry_run() {
            let dir = self.out.join("tmp-dry-run");
            t!(std::fs::create_dir_all(&dir));
            self.out = dir;
        }

        self.hosts = flags_host
            .map(|TargetSelectionList(list)| list)
            .or_else(|| {
                host.map(|h| h.into_iter().map(|s| TargetSelection::from_user(&s)).collect())
            })
            .unwrap_or_else(|| vec![self.host_target.clone()]);

        self.targets = flags_target
            .map(|TargetSelectionList(list)| list)
            .or_else(|| {
                target.map(|t| t.into_iter().map(|s| TargetSelection::from_user(&s)).collect())
            })
            .unwrap_or_else(|| self.hosts.clone());

        self.nodejs = nodejs.map(PathBuf::from);
        self.npm = npm.map(PathBuf::from);
        self.gdb = gdb.map(PathBuf::from);
        self.lldb = lldb.map(PathBuf::from);
        self.python = python.map(PathBuf::from);
        self.reuse = reuse.map(PathBuf::from);

        self.submodules = submodules;
        self.android_ndk = android_ndk;
        self.bootstrap_cache_path = bootstrap_cache_path;
        self.tools = tools;

        set(&mut self.low_priority, low_priority);
        set(&mut self.compiler_docs, compiler_docs);
        set(&mut self.library_docs_private_items, library_docs_private_items);
        set(&mut self.docs_minification, docs_minification);
        set(&mut self.docs, docs);
        set(&mut self.locked_deps, locked_deps);
        set(&mut self.full_bootstrap, full_bootstrap);
        set(&mut self.extended, extended);
        set(&mut self.tool, tool);
        set(&mut self.verbose, verbose);
        set(&mut self.sanitizers, sanitizers);
        set(&mut self.profiler, profiler);
        set(&mut self.cargo_native_static, cargo_native_static);
        set(&mut self.configure_args, configure_args);
        set(&mut self.local_rebuild, local_rebuild);
        set(&mut self.print_step_timings, print_step_timings);
        set(&mut self.print_step_rusage, print_step_rusage);
        self.patch_binaries_for_nix = patch_binaries_for_nix;

        self.vendor = vendor.unwrap_or(
            self.rust_info.is_from_tarball()
                && self.src.join("vendor").exists()
                && self.src.join(".cargo/config.toml").exists(),
        );

        self.initial_rustfmt = rustfmt.or_else(|| self.maybe_download_rustfmt());

        self.optimized_compiler_builtins =
            optimized_compiler_builtins.unwrap_or(self.channel != "dev");
        self.compiletest_diff_tool = compiletest_diff_tool;
        self.compiletest_use_stage0_libtest = compiletest_use_stage0_libtest.unwrap_or(true);
        self.tidy_extra_checks = tidy_extra_checks;

        let download_rustc = self.download_rustc_commit.is_some();
        self.explicit_stage_from_config = test_stage.is_some()
            || build_stage.is_some()
            || doc_stage.is_some()
            || dist_stage.is_some()
            || install_stage.is_some()
            || check_stage.is_some()
            || bench_stage.is_some();

        self.stage = match self.cmd {
            Subcommand::Check { .. } | Subcommand::Clippy { .. } | Subcommand::Fix => {
                flags_stage.or(check_stage).unwrap_or(1)
            }
            // `download-rustc` only has a speed-up for stage2 builds. Default to stage2 unless explicitly overridden.
            Subcommand::Doc { .. } => {
                flags_stage.or(doc_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Build => {
                flags_stage.or(build_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Test { .. } | Subcommand::Miri { .. } => {
                flags_stage.or(test_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Bench { .. } => flags_stage.or(bench_stage).unwrap_or(2),
            Subcommand::Dist => flags_stage.or(dist_stage).unwrap_or(2),
            Subcommand::Install => flags_stage.or(install_stage).unwrap_or(2),
            Subcommand::Perf { .. } => flags_stage.unwrap_or(1),
            // These are all bootstrap tools, which don't depend on the compiler.
            // The stage we pass shouldn't matter, but use 0 just in case.
            Subcommand::Clean { .. }
            | Subcommand::Run { .. }
            | Subcommand::Setup { .. }
            | Subcommand::Format { .. }
            | Subcommand::Vendor { .. } => flags_stage.unwrap_or(0),
        };
    }

    #[cfg(test)]
    pub fn check_stage0_version(&self, _program_path: &Path, _component_name: &'static str) {
        use std::path::Path;
    }

    /// check rustc/cargo version is same or lower with 1 apart from the building one
    #[cfg(not(test))]
    pub fn check_stage0_version(&self, program_path: &Path, component_name: &'static str) {
        use std::fs;

        use build_helper::util::fail;

        if self.dry_run() {
            return;
        }

        let stage0_output =
            command(program_path).arg("--version").run_capture_stdout(self).stdout();
        let mut stage0_output = stage0_output.lines().next().unwrap().split(' ');

        let stage0_name = stage0_output.next().unwrap();
        if stage0_name != component_name {
            fail(&format!(
                "Expected to find {component_name} at {} but it claims to be {stage0_name}",
                program_path.display()
            ));
        }

        let stage0_version =
            semver::Version::parse(stage0_output.next().unwrap().split('-').next().unwrap().trim())
                .unwrap();
        let source_version = semver::Version::parse(
            fs::read_to_string(self.src.join("src/version")).unwrap().trim(),
        )
        .unwrap();
        if !(source_version == stage0_version
            || (source_version.major == stage0_version.major
                && (source_version.minor == stage0_version.minor
                    || source_version.minor == stage0_version.minor + 1)))
        {
            let prev_version = format!("{}.{}.x", source_version.major, source_version.minor - 1);
            fail(&format!(
                "Unexpected {component_name} version: {stage0_version}, we should use {prev_version}/{source_version} to build source with {source_version}"
            ));
        }
    }

    #[cfg(test)]
    pub(crate) fn download_beta_toolchain(&self) {}

    #[cfg(not(test))]
    pub(crate) fn download_beta_toolchain(&self) {
        use crate::core::download::DownloadSource;

        self.verbose(|| println!("downloading stage0 beta artifacts"));

        let date = &self.stage0_metadata.compiler.date;
        let version = &self.stage0_metadata.compiler.version;
        let extra_components = ["cargo"];

        let download_beta_component = |config: &Config, filename, prefix: &_, date: &_| {
            config.download_component(DownloadSource::Dist, filename, prefix, date, "stage0")
        };

        self.download_toolchain(
            version,
            "stage0",
            date,
            &extra_components,
            download_beta_component,
        );
    }

    #[cfg(test)]
    pub(crate) fn maybe_download_rustfmt(&self) -> Option<PathBuf> {
        Some(PathBuf::new())
    }

    /// NOTE: rustfmt is a completely different toolchain than the bootstrap compiler, so it can't
    /// reuse target directories or artifacts
    #[cfg(not(test))]
    pub(crate) fn maybe_download_rustfmt(&self) -> Option<PathBuf> {
        use build_helper::stage0_parser::VersionMetadata;

        use crate::core::download::{DownloadSource, path_is_dylib};
        use crate::fs;
        use crate::utils::build_stamp::BuildStamp;

        if self.dry_run() {
            return Some(PathBuf::new());
        }

        let VersionMetadata { date, version } = self.stage0_metadata.rustfmt.as_ref()?;
        let channel = format!("{version}-{date}");

        let host = self.host_target;
        let bin_root = self.out.join(host).join("rustfmt");
        let rustfmt_path = bin_root.join("bin").join(exe("rustfmt", host));
        let rustfmt_stamp = BuildStamp::new(&bin_root).with_prefix("rustfmt").add_stamp(channel);
        if rustfmt_path.exists() && rustfmt_stamp.is_up_to_date() {
            return Some(rustfmt_path);
        }

        self.download_component(
            DownloadSource::Dist,
            format!("rustfmt-{version}-{build}.tar.xz", build = host.triple),
            "rustfmt-preview",
            date,
            "rustfmt",
        );
        self.download_component(
            DownloadSource::Dist,
            format!("rustc-{version}-{build}.tar.xz", build = host.triple),
            "rustc",
            date,
            "rustfmt",
        );

        if self.should_fix_bins_and_dylibs() {
            self.fix_bin_or_dylib(&bin_root.join("bin").join("rustfmt"));
            self.fix_bin_or_dylib(&bin_root.join("bin").join("cargo-fmt"));
            let lib_dir = bin_root.join("lib");
            for lib in t!(fs::read_dir(&lib_dir), lib_dir.display().to_string()) {
                let lib = t!(lib);
                if path_is_dylib(&lib.path()) {
                    self.fix_bin_or_dylib(&lib.path());
                }
            }
        }

        t!(rustfmt_stamp.write());
        Some(rustfmt_path)
    }
}
