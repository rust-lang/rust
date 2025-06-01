use std::env;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use super::{Builder, Kind};
use crate::core::build_steps::tool::SourceType;
use crate::core::build_steps::{compile, test};
use crate::core::config::SplitDebuginfo;
use crate::core::config::flags::Color;
use crate::utils::build_stamp;
use crate::utils::helpers::{self, LldThreads, check_cfg_arg, linker_args, linker_flags};
use crate::{
    BootstrapCommand, CLang, Compiler, Config, DocTests, DryRun, EXTRA_CHECK_CFGS, GitRepo, Mode,
    TargetSelection, command, prepare_behaviour_dump_dir, t,
};

/// Represents flag values in `String` form with whitespace delimiter to pass it to the compiler
/// later.
///
/// `-Z crate-attr` flags will be applied recursively on the target code using the
/// `rustc_parse::parser::Parser`. See `rustc_builtin_macros::cmdline_attrs::inject` for more
/// information.
#[derive(Debug, Clone)]
struct Rustflags(String, TargetSelection);

impl Rustflags {
    fn new(target: TargetSelection) -> Rustflags {
        let mut ret = Rustflags(String::new(), target);
        ret.propagate_cargo_env("RUSTFLAGS");
        ret
    }

    /// By default, cargo will pick up on various variables in the environment. However, bootstrap
    /// reuses those variables to pass additional flags to rustdoc, so by default they get
    /// overridden. Explicitly add back any previous value in the environment.
    ///
    /// `prefix` is usually `RUSTFLAGS` or `RUSTDOCFLAGS`.
    fn propagate_cargo_env(&mut self, prefix: &str) {
        // Inherit `RUSTFLAGS` by default ...
        self.env(prefix);

        // ... and also handle target-specific env RUSTFLAGS if they're configured.
        let target_specific = format!("CARGO_TARGET_{}_{}", crate::envify(&self.1.triple), prefix);
        self.env(&target_specific);
    }

    fn env(&mut self, env: &str) {
        if let Ok(s) = env::var(env) {
            for part in s.split(' ') {
                self.arg(part);
            }
        }
    }

    fn arg(&mut self, arg: &str) -> &mut Self {
        assert_eq!(arg.split(' ').count(), 1);
        if !self.0.is_empty() {
            self.0.push(' ');
        }
        self.0.push_str(arg);
        self
    }
}

/// Flags that are passed to the `rustc` shim binary. These flags will only be applied when
/// compiling host code, i.e. when `--target` is unset.
#[derive(Debug, Default)]
struct HostFlags {
    rustc: Vec<String>,
}

impl HostFlags {
    const SEPARATOR: &'static str = " ";

    /// Adds a host rustc flag.
    fn arg<S: Into<String>>(&mut self, flag: S) {
        let value = flag.into().trim().to_string();
        assert!(!value.contains(Self::SEPARATOR));
        self.rustc.push(value);
    }

    /// Encodes all the flags into a single string.
    fn encode(self) -> String {
        self.rustc.join(Self::SEPARATOR)
    }
}

#[derive(Debug)]
pub struct Cargo {
    command: BootstrapCommand,
    args: Vec<OsString>,
    compiler: Compiler,
    target: TargetSelection,
    rustflags: Rustflags,
    rustdocflags: Rustflags,
    hostflags: HostFlags,
    allow_features: String,
    release_build: bool,
}

impl Cargo {
    /// Calls [`Builder::cargo`] and [`Cargo::configure_linker`] to prepare an invocation of `cargo`
    /// to be run.
    pub fn new(
        builder: &Builder<'_>,
        compiler: Compiler,
        mode: Mode,
        source_type: SourceType,
        target: TargetSelection,
        cmd_kind: Kind,
    ) -> Cargo {
        let mut cargo = builder.cargo(compiler, mode, source_type, target, cmd_kind);

        match cmd_kind {
            // No need to configure the target linker for these command types.
            Kind::Clean | Kind::Check | Kind::Suggest | Kind::Format | Kind::Setup => {}
            _ => {
                cargo.configure_linker(builder);
            }
        }

        cargo
    }

    pub fn release_build(&mut self, release_build: bool) {
        self.release_build = release_build;
    }

    pub fn compiler(&self) -> Compiler {
        self.compiler
    }

    pub fn into_cmd(self) -> BootstrapCommand {
        self.into()
    }

    /// Same as [`Cargo::new`] except this one doesn't configure the linker with
    /// [`Cargo::configure_linker`].
    pub fn new_for_mir_opt_tests(
        builder: &Builder<'_>,
        compiler: Compiler,
        mode: Mode,
        source_type: SourceType,
        target: TargetSelection,
        cmd_kind: Kind,
    ) -> Cargo {
        builder.cargo(compiler, mode, source_type, target, cmd_kind)
    }

    pub fn rustdocflag(&mut self, arg: &str) -> &mut Cargo {
        self.rustdocflags.arg(arg);
        self
    }

    pub fn rustflag(&mut self, arg: &str) -> &mut Cargo {
        self.rustflags.arg(arg);
        self
    }

    pub fn arg(&mut self, arg: impl AsRef<OsStr>) -> &mut Cargo {
        self.args.push(arg.as_ref().into());
        self
    }

    pub fn args<I, S>(&mut self, args: I) -> &mut Cargo
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        for arg in args {
            self.arg(arg.as_ref());
        }
        self
    }

    /// Add an env var to the cargo command instance. Note that `RUSTFLAGS`/`RUSTDOCFLAGS` must go
    /// through [`Cargo::rustdocflags`] and [`Cargo::rustflags`] because inconsistent `RUSTFLAGS`
    /// and `RUSTDOCFLAGS` usages will trigger spurious rebuilds.
    pub fn env(&mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> &mut Cargo {
        assert_ne!(key.as_ref(), "RUSTFLAGS");
        assert_ne!(key.as_ref(), "RUSTDOCFLAGS");
        self.command.env(key.as_ref(), value.as_ref());
        self
    }

    pub fn add_rustc_lib_path(&mut self, builder: &Builder<'_>) {
        builder.add_rustc_lib_path(self.compiler, &mut self.command);
    }

    pub fn current_dir(&mut self, dir: &Path) -> &mut Cargo {
        self.command.current_dir(dir);
        self
    }

    /// Adds nightly-only features that this invocation is allowed to use.
    ///
    /// By default, all nightly features are allowed. Once this is called, it will be restricted to
    /// the given set.
    pub fn allow_features(&mut self, features: &str) -> &mut Cargo {
        if !self.allow_features.is_empty() {
            self.allow_features.push(',');
        }
        self.allow_features.push_str(features);
        self
    }

    // FIXME(onur-ozkan): Add coverage to make sure modifications to this function
    // doesn't cause cache invalidations (e.g., #130108).
    fn configure_linker(&mut self, builder: &Builder<'_>) -> &mut Cargo {
        let target = self.target;
        let compiler = self.compiler;

        // Dealing with rpath here is a little special, so let's go into some
        // detail. First off, `-rpath` is a linker option on Unix platforms
        // which adds to the runtime dynamic loader path when looking for
        // dynamic libraries. We use this by default on Unix platforms to ensure
        // that our nightlies behave the same on Windows, that is they work out
        // of the box. This can be disabled by setting `rpath = false` in `[rust]`
        // table of `bootstrap.toml`
        //
        // Ok, so the astute might be wondering "why isn't `-C rpath` used
        // here?" and that is indeed a good question to ask. This codegen
        // option is the compiler's current interface to generating an rpath.
        // Unfortunately it doesn't quite suffice for us. The flag currently
        // takes no value as an argument, so the compiler calculates what it
        // should pass to the linker as `-rpath`. This unfortunately is based on
        // the **compile time** directory structure which when building with
        // Cargo will be very different than the runtime directory structure.
        //
        // All that's a really long winded way of saying that if we use
        // `-Crpath` then the executables generated have the wrong rpath of
        // something like `$ORIGIN/deps` when in fact the way we distribute
        // rustc requires the rpath to be `$ORIGIN/../lib`.
        //
        // So, all in all, to set up the correct rpath we pass the linker
        // argument manually via `-C link-args=-Wl,-rpath,...`. Plus isn't it
        // fun to pass a flag to a tool to pass a flag to pass a flag to a tool
        // to change a flag in a binary?
        if builder.config.rpath_enabled(target) && helpers::use_host_linker(target) {
            let libdir = builder.sysroot_libdir_relative(compiler).to_str().unwrap();
            let rpath = if target.contains("apple") {
                // Note that we need to take one extra step on macOS to also pass
                // `-Wl,-instal_name,@rpath/...` to get things to work right. To
                // do that we pass a weird flag to the compiler to get it to do
                // so. Note that this is definitely a hack, and we should likely
                // flesh out rpath support more fully in the future.
                self.rustflags.arg("-Zosx-rpath-install-name");
                Some(format!("-Wl,-rpath,@loader_path/../{libdir}"))
            } else if !target.is_windows()
                && !target.contains("cygwin")
                && !target.contains("aix")
                && !target.contains("xous")
            {
                self.rustflags.arg("-Clink-args=-Wl,-z,origin");
                Some(format!("-Wl,-rpath,$ORIGIN/../{libdir}"))
            } else {
                None
            };
            if let Some(rpath) = rpath {
                self.rustflags.arg(&format!("-Clink-args={rpath}"));
            }
        }

        for arg in linker_args(builder, compiler.host, LldThreads::Yes) {
            self.hostflags.arg(&arg);
        }

        if let Some(target_linker) = builder.linker(target) {
            let target = crate::envify(&target.triple);
            self.command.env(format!("CARGO_TARGET_{target}_LINKER"), target_linker);
        }
        // We want to set -Clinker using Cargo, therefore we only call `linker_flags` and not
        // `linker_args` here.
        for flag in linker_flags(builder, target, LldThreads::Yes) {
            self.rustflags.arg(&flag);
        }
        for arg in linker_args(builder, target, LldThreads::Yes) {
            self.rustdocflags.arg(&arg);
        }

        if !builder.config.dry_run()
            && builder.cc.borrow()[&target].args().iter().any(|arg| arg == "-gz")
        {
            self.rustflags.arg("-Clink-arg=-gz");
        }

        // Ignore linker warnings for now. These are complicated to fix and don't affect the build.
        // FIXME: we should really investigate these...
        self.rustflags.arg("-Alinker-messages");

        // Throughout the build Cargo can execute a number of build scripts
        // compiling C/C++ code and we need to pass compilers, archivers, flags, etc
        // obtained previously to those build scripts.
        // Build scripts use either the `cc` crate or `configure/make` so we pass
        // the options through environment variables that are fetched and understood by both.
        //
        // FIXME: the guard against msvc shouldn't need to be here
        if target.is_msvc() {
            if let Some(ref cl) = builder.config.llvm_clang_cl {
                // FIXME: There is a bug in Clang 18 when building for ARM64:
                // https://github.com/llvm/llvm-project/pull/81849. This is
                // fixed in LLVM 19, but can't be backported.
                if !target.starts_with("aarch64") && !target.starts_with("arm64ec") {
                    self.command.env("CC", cl).env("CXX", cl);
                }
            }
        } else {
            let ccache = builder.config.ccache.as_ref();
            let ccacheify = |s: &Path| {
                let ccache = match ccache {
                    Some(ref s) => s,
                    None => return s.display().to_string(),
                };
                // FIXME: the cc-rs crate only recognizes the literal strings
                // `ccache` and `sccache` when doing caching compilations, so we
                // mirror that here. It should probably be fixed upstream to
                // accept a new env var or otherwise work with custom ccache
                // vars.
                match &ccache[..] {
                    "ccache" | "sccache" => format!("{} {}", ccache, s.display()),
                    _ => s.display().to_string(),
                }
            };
            let triple_underscored = target.triple.replace('-', "_");
            let cc = ccacheify(&builder.cc(target));
            self.command.env(format!("CC_{triple_underscored}"), &cc);

            // Extend `CXXFLAGS_$TARGET` with our extra flags.
            let env = format!("CFLAGS_{triple_underscored}");
            let mut cflags =
                builder.cc_unhandled_cflags(target, GitRepo::Rustc, CLang::C).join(" ");
            if let Ok(var) = std::env::var(&env) {
                cflags.push(' ');
                cflags.push_str(&var);
            }
            self.command.env(env, &cflags);

            if let Some(ar) = builder.ar(target) {
                let ranlib = format!("{} s", ar.display());
                self.command
                    .env(format!("AR_{triple_underscored}"), ar)
                    .env(format!("RANLIB_{triple_underscored}"), ranlib);
            }

            if let Ok(cxx) = builder.cxx(target) {
                let cxx = ccacheify(&cxx);
                self.command.env(format!("CXX_{triple_underscored}"), &cxx);

                // Extend `CXXFLAGS_$TARGET` with our extra flags.
                let env = format!("CXXFLAGS_{triple_underscored}");
                let mut cxxflags =
                    builder.cc_unhandled_cflags(target, GitRepo::Rustc, CLang::Cxx).join(" ");
                if let Ok(var) = std::env::var(&env) {
                    cxxflags.push(' ');
                    cxxflags.push_str(&var);
                }
                self.command.env(&env, cxxflags);
            }
        }

        self
    }
}

impl From<Cargo> for BootstrapCommand {
    fn from(mut cargo: Cargo) -> BootstrapCommand {
        if cargo.release_build {
            cargo.args.insert(0, "--release".into());
        }

        cargo.command.args(cargo.args);

        let rustflags = &cargo.rustflags.0;
        if !rustflags.is_empty() {
            cargo.command.env("RUSTFLAGS", rustflags);
        }

        let rustdocflags = &cargo.rustdocflags.0;
        if !rustdocflags.is_empty() {
            cargo.command.env("RUSTDOCFLAGS", rustdocflags);
        }

        let encoded_hostflags = cargo.hostflags.encode();
        if !encoded_hostflags.is_empty() {
            cargo.command.env("RUSTC_HOST_FLAGS", encoded_hostflags);
        }

        if !cargo.allow_features.is_empty() {
            cargo.command.env("RUSTC_ALLOW_FEATURES", cargo.allow_features);
        }

        cargo.command
    }
}

impl Builder<'_> {
    /// Like [`Builder::cargo`], but only passes flags that are valid for all commands.
    pub fn bare_cargo(
        &self,
        compiler: Compiler,
        mode: Mode,
        target: TargetSelection,
        cmd_kind: Kind,
    ) -> BootstrapCommand {
        let mut cargo = match cmd_kind {
            Kind::Clippy => {
                let mut cargo = self.cargo_clippy_cmd(compiler);
                cargo.arg(cmd_kind.as_str());
                cargo
            }
            Kind::MiriSetup => {
                let mut cargo = self.cargo_miri_cmd(compiler);
                cargo.arg("miri").arg("setup");
                cargo
            }
            Kind::MiriTest => {
                let mut cargo = self.cargo_miri_cmd(compiler);
                cargo.arg("miri").arg("test");
                cargo
            }
            _ => {
                let mut cargo = command(&self.initial_cargo);
                cargo.arg(cmd_kind.as_str());
                cargo
            }
        };

        // Run cargo from the source root so it can find .cargo/config.
        // This matters when using vendoring and the working directory is outside the repository.
        cargo.current_dir(&self.src);

        let out_dir = self.stage_out(compiler, mode);
        cargo.env("CARGO_TARGET_DIR", &out_dir);

        // Found with `rg "init_env_logger\("`. If anyone uses `init_env_logger`
        // from out of tree it shouldn't matter, since x.py is only used for
        // building in-tree.
        let color_logs = ["RUSTDOC_LOG_COLOR", "RUSTC_LOG_COLOR", "RUST_LOG_COLOR"];
        match self.build.config.color {
            Color::Always => {
                cargo.arg("--color=always");
                for log in &color_logs {
                    cargo.env(log, "always");
                }
            }
            Color::Never => {
                cargo.arg("--color=never");
                for log in &color_logs {
                    cargo.env(log, "never");
                }
            }
            Color::Auto => {} // nothing to do
        }

        if cmd_kind != Kind::Install {
            cargo.arg("--target").arg(target.rustc_target_arg());
        } else {
            assert_eq!(target, compiler.host);
        }

        // Remove make-related flags to ensure Cargo can correctly set things up
        cargo.env_remove("MAKEFLAGS");
        cargo.env_remove("MFLAGS");

        cargo
    }

    /// This will create a [`BootstrapCommand`] that represents a pending execution of cargo. This
    /// cargo will be configured to use `compiler` as the actual rustc compiler, its output will be
    /// scoped by `mode`'s output directory, it will pass the `--target` flag for the specified
    /// `target`, and will be executing the Cargo command `cmd`. `cmd` can be `miri-cmd` for
    /// commands to be run with Miri.
    fn cargo(
        &self,
        compiler: Compiler,
        mode: Mode,
        source_type: SourceType,
        target: TargetSelection,
        cmd_kind: Kind,
    ) -> Cargo {
        let mut cargo = self.bare_cargo(compiler, mode, target, cmd_kind);
        let out_dir = self.stage_out(compiler, mode);

        let mut hostflags = HostFlags::default();

        // Codegen backends are not yet tracked by -Zbinary-dep-depinfo,
        // so we need to explicitly clear out if they've been updated.
        for backend in self.codegen_backends(compiler) {
            build_stamp::clear_if_dirty(self, &out_dir, &backend);
        }

        if cmd_kind == Kind::Doc {
            let my_out = match mode {
                // This is the intended out directory for compiler documentation.
                Mode::Rustc | Mode::ToolRustc => self.compiler_doc_out(target),
                Mode::Std => {
                    if self.config.cmd.json() {
                        out_dir.join(target).join("json-doc")
                    } else {
                        out_dir.join(target).join("doc")
                    }
                }
                _ => panic!("doc mode {mode:?} not expected"),
            };
            let rustdoc = self.rustdoc(compiler);
            build_stamp::clear_if_dirty(self, &my_out, &rustdoc);
        }

        let profile_var = |name: &str| cargo_profile_var(name, &self.config);

        // See comment in rustc_llvm/build.rs for why this is necessary, largely llvm-config
        // needs to not accidentally link to libLLVM in stage0/lib.
        cargo.env("REAL_LIBRARY_PATH_VAR", helpers::dylib_path_var());
        if let Some(e) = env::var_os(helpers::dylib_path_var()) {
            cargo.env("REAL_LIBRARY_PATH", e);
        }

        // Set a flag for `check`/`clippy`/`fix`, so that certain build
        // scripts can do less work (i.e. not building/requiring LLVM).
        if matches!(cmd_kind, Kind::Check | Kind::Clippy | Kind::Fix) {
            // If we've not yet built LLVM, or it's stale, then bust
            // the rustc_llvm cache. That will always work, even though it
            // may mean that on the next non-check build we'll need to rebuild
            // rustc_llvm. But if LLVM is stale, that'll be a tiny amount
            // of work comparatively, and we'd likely need to rebuild it anyway,
            // so that's okay.
            if crate::core::build_steps::llvm::prebuilt_llvm_config(self, target, false)
                .should_build()
            {
                cargo.env("RUST_CHECK", "1");
            }
        }

        let stage = if compiler.stage == 0 && self.local_rebuild {
            // Assume the local-rebuild rustc already has stage1 features.
            1
        } else {
            compiler.stage
        };

        // We synthetically interpret a stage0 compiler used to build tools as a
        // "raw" compiler in that it's the exact snapshot we download. Normally
        // the stage0 build means it uses libraries build by the stage0
        // compiler, but for tools we just use the precompiled libraries that
        // we've downloaded
        let use_snapshot = mode == Mode::ToolBootstrap;
        assert!(!use_snapshot || stage == 0 || self.local_rebuild);

        let maybe_sysroot = self.sysroot(compiler);
        let sysroot = if use_snapshot { self.rustc_snapshot_sysroot() } else { &maybe_sysroot };
        let libdir = self.rustc_libdir(compiler);

        let sysroot_str = sysroot.as_os_str().to_str().expect("sysroot should be UTF-8");
        if self.is_verbose() && !matches!(self.config.dry_run, DryRun::SelfCheck) {
            println!("using sysroot {sysroot_str}");
        }

        let mut rustflags = Rustflags::new(target);
        if stage != 0 {
            if let Ok(s) = env::var("CARGOFLAGS_NOT_BOOTSTRAP") {
                cargo.args(s.split_whitespace());
            }
            rustflags.env("RUSTFLAGS_NOT_BOOTSTRAP");
        } else {
            if let Ok(s) = env::var("CARGOFLAGS_BOOTSTRAP") {
                cargo.args(s.split_whitespace());
            }
            rustflags.env("RUSTFLAGS_BOOTSTRAP");
            rustflags.arg("--cfg=bootstrap");
        }

        if cmd_kind == Kind::Clippy {
            // clippy overwrites sysroot if we pass it to cargo.
            // Pass it directly to clippy instead.
            // NOTE: this can't be fixed in clippy because we explicitly don't set `RUSTC`,
            // so it has no way of knowing the sysroot.
            rustflags.arg("--sysroot");
            rustflags.arg(sysroot_str);
        }

        let use_new_symbol_mangling = match self.config.rust_new_symbol_mangling {
            Some(setting) => {
                // If an explicit setting is given, use that
                setting
            }
            None => {
                if mode == Mode::Std {
                    // The standard library defaults to the legacy scheme
                    false
                } else {
                    // The compiler and tools default to the new scheme
                    true
                }
            }
        };

        // By default, windows-rs depends on a native library that doesn't get copied into the
        // sysroot. Passing this cfg enables raw-dylib support instead, which makes the native
        // library unnecessary. This can be removed when windows-rs enables raw-dylib
        // unconditionally.
        if let Mode::Rustc | Mode::ToolRustc | Mode::ToolBootstrap = mode {
            rustflags.arg("--cfg=windows_raw_dylib");
        }

        if use_new_symbol_mangling {
            rustflags.arg("-Csymbol-mangling-version=v0");
        } else {
            rustflags.arg("-Csymbol-mangling-version=legacy");
        }

        // FIXME: the following components don't build with `-Zrandomize-layout` yet:
        // - rust-analyzer, due to the rowan crate
        // so we exclude an entire category of steps here due to lack of fine-grained control over
        // rustflags.
        if self.config.rust_randomize_layout && mode != Mode::ToolRustc {
            rustflags.arg("-Zrandomize-layout");
        }

        // Enable compile-time checking of `cfg` names, values and Cargo `features`.
        //
        // Note: `std`, `alloc` and `core` imports some dependencies by #[path] (like
        // backtrace, core_simd, std_float, ...), those dependencies have their own
        // features but cargo isn't involved in the #[path] process and so cannot pass the
        // complete list of features, so for that reason we don't enable checking of
        // features for std crates.
        if mode == Mode::Std {
            rustflags.arg("--check-cfg=cfg(feature,values(any()))");
        }

        // Add extra cfg not defined in/by rustc
        //
        // Note: Although it would seems that "-Zunstable-options" to `rustflags` is useless as
        // cargo would implicitly add it, it was discover that sometimes bootstrap only use
        // `rustflags` without `cargo` making it required.
        rustflags.arg("-Zunstable-options");
        for (restricted_mode, name, values) in EXTRA_CHECK_CFGS {
            if restricted_mode.is_none() || *restricted_mode == Some(mode) {
                rustflags.arg(&check_cfg_arg(name, *values));
            }
        }

        // FIXME(rust-lang/cargo#5754) we shouldn't be using special command arguments
        // to the host invocation here, but rather Cargo should know what flags to pass rustc
        // itself.
        if stage == 0 {
            hostflags.arg("--cfg=bootstrap");
        }
        // Cargo doesn't pass RUSTFLAGS to proc_macros:
        // https://github.com/rust-lang/cargo/issues/4423
        // Thus, if we are on stage 0, we explicitly set `--cfg=bootstrap`.
        // We also declare that the flag is expected, which we need to do to not
        // get warnings about it being unexpected.
        hostflags.arg("-Zunstable-options");
        hostflags.arg("--check-cfg=cfg(bootstrap)");

        // FIXME: It might be better to use the same value for both `RUSTFLAGS` and `RUSTDOCFLAGS`,
        // but this breaks CI. At the very least, stage0 `rustdoc` needs `--cfg bootstrap`. See
        // #71458.
        let mut rustdocflags = rustflags.clone();
        rustdocflags.propagate_cargo_env("RUSTDOCFLAGS");
        if stage == 0 {
            rustdocflags.env("RUSTDOCFLAGS_BOOTSTRAP");
        } else {
            rustdocflags.env("RUSTDOCFLAGS_NOT_BOOTSTRAP");
        }

        if let Ok(s) = env::var("CARGOFLAGS") {
            cargo.args(s.split_whitespace());
        }

        match mode {
            Mode::Std | Mode::ToolBootstrap | Mode::ToolStd => {}
            Mode::Rustc | Mode::Codegen | Mode::ToolRustc => {
                // Build proc macros both for the host and the target unless proc-macros are not
                // supported by the target.
                if target != compiler.host && cmd_kind != Kind::Check {
                    let mut rustc_cmd = command(self.rustc(compiler));
                    self.add_rustc_lib_path(compiler, &mut rustc_cmd);

                    let error = rustc_cmd
                        .arg("--target")
                        .arg(target.rustc_target_arg())
                        .arg("--print=file-names")
                        .arg("--crate-type=proc-macro")
                        .arg("-")
                        .run_capture(self)
                        .stderr();

                    let not_supported = error
                        .lines()
                        .any(|line| line.contains("unsupported crate type `proc-macro`"));
                    if !not_supported {
                        cargo.arg("-Zdual-proc-macros");
                        rustflags.arg("-Zdual-proc-macros");
                    }
                }
            }
        }

        // This tells Cargo (and in turn, rustc) to output more complete
        // dependency information.  Most importantly for bootstrap, this
        // includes sysroot artifacts, like libstd, which means that we don't
        // need to track those in bootstrap (an error prone process!). This
        // feature is currently unstable as there may be some bugs and such, but
        // it represents a big improvement in bootstrap's reliability on
        // rebuilds, so we're using it here.
        //
        // For some additional context, see #63470 (the PR originally adding
        // this), as well as #63012 which is the tracking issue for this
        // feature on the rustc side.
        cargo.arg("-Zbinary-dep-depinfo");
        let allow_features = match mode {
            Mode::ToolBootstrap | Mode::ToolStd => {
                // Restrict the allowed features so we don't depend on nightly
                // accidentally.
                //
                // binary-dep-depinfo is used by bootstrap itself for all
                // compilations.
                //
                // Lots of tools depend on proc_macro2 and proc-macro-error.
                // Those have build scripts which assume nightly features are
                // available if the `rustc` version is "nighty" or "dev". See
                // bin/rustc.rs for why that is a problem. Instead of labeling
                // those features for each individual tool that needs them,
                // just blanket allow them here.
                //
                // If this is ever removed, be sure to add something else in
                // its place to keep the restrictions in place (or make a way
                // to unset RUSTC_BOOTSTRAP).
                "binary-dep-depinfo,proc_macro_span,proc_macro_span_shrink,proc_macro_diagnostic"
                    .to_string()
            }
            Mode::Std | Mode::Rustc | Mode::Codegen | Mode::ToolRustc => String::new(),
        };

        cargo.arg("-j").arg(self.jobs().to_string());

        // Make cargo emit diagnostics relative to the rustc src dir.
        cargo.arg(format!("-Zroot-dir={}", self.src.display()));

        // FIXME: Temporary fix for https://github.com/rust-lang/cargo/issues/3005
        // Force cargo to output binaries with disambiguating hashes in the name
        let mut metadata = if compiler.stage == 0 {
            // Treat stage0 like a special channel, whether it's a normal prior-
            // release rustc or a local rebuild with the same version, so we
            // never mix these libraries by accident.
            "bootstrap".to_string()
        } else {
            self.config.channel.to_string()
        };
        // We want to make sure that none of the dependencies between
        // std/test/rustc unify with one another. This is done for weird linkage
        // reasons but the gist of the problem is that if librustc, libtest, and
        // libstd all depend on libc from crates.io (which they actually do) we
        // want to make sure they all get distinct versions. Things get really
        // weird if we try to unify all these dependencies right now, namely
        // around how many times the library is linked in dynamic libraries and
        // such. If rustc were a static executable or if we didn't ship dylibs
        // this wouldn't be a problem, but we do, so it is. This is in general
        // just here to make sure things build right. If you can remove this and
        // things still build right, please do!
        match mode {
            Mode::Std => metadata.push_str("std"),
            // When we're building rustc tools, they're built with a search path
            // that contains things built during the rustc build. For example,
            // bitflags is built during the rustc build, and is a dependency of
            // rustdoc as well. We're building rustdoc in a different target
            // directory, though, which means that Cargo will rebuild the
            // dependency. When we go on to build rustdoc, we'll look for
            // bitflags, and find two different copies: one built during the
            // rustc step and one that we just built. This isn't always a
            // problem, somehow -- not really clear why -- but we know that this
            // fixes things.
            Mode::ToolRustc => metadata.push_str("tool-rustc"),
            // Same for codegen backends.
            Mode::Codegen => metadata.push_str("codegen"),
            _ => {}
        }
        // `rustc_driver`'s version number is always `0.0.0`, which can cause linker search path
        // problems on side-by-side installs because we don't include the version number of the
        // `rustc_driver` being built. This can cause builds of different version numbers to produce
        // `librustc_driver*.so` artifacts that end up with identical filename hashes.
        metadata.push_str(&self.version);

        cargo.env("__CARGO_DEFAULT_LIB_METADATA", &metadata);

        if cmd_kind == Kind::Clippy {
            rustflags.arg("-Zforce-unstable-if-unmarked");
        }

        rustflags.arg("-Zmacro-backtrace");

        let want_rustdoc = self.doc_tests != DocTests::No;

        // Clear the output directory if the real rustc we're using has changed;
        // Cargo cannot detect this as it thinks rustc is bootstrap/debug/rustc.
        //
        // Avoid doing this during dry run as that usually means the relevant
        // compiler is not yet linked/copied properly.
        //
        // Only clear out the directory if we're compiling std; otherwise, we
        // should let Cargo take care of things for us (via depdep info)
        if !self.config.dry_run() && mode == Mode::Std && cmd_kind == Kind::Build {
            build_stamp::clear_if_dirty(self, &out_dir, &self.rustc(compiler));
        }

        let rustdoc_path = match cmd_kind {
            Kind::Doc | Kind::Test | Kind::MiriTest => self.rustdoc(compiler),
            _ => PathBuf::from("/path/to/nowhere/rustdoc/not/required"),
        };

        // Customize the compiler we're running. Specify the compiler to cargo
        // as our shim and then pass it some various options used to configure
        // how the actual compiler itself is called.
        //
        // These variables are primarily all read by
        // src/bootstrap/bin/{rustc.rs,rustdoc.rs}
        cargo
            .env("RUSTBUILD_NATIVE_DIR", self.native_dir(target))
            .env("RUSTC_REAL", self.rustc(compiler))
            .env("RUSTC_STAGE", stage.to_string())
            .env("RUSTC_SYSROOT", sysroot)
            .env("RUSTC_LIBDIR", libdir)
            .env("RUSTDOC", self.bootstrap_out.join("rustdoc"))
            .env("RUSTDOC_REAL", rustdoc_path)
            .env("RUSTC_ERROR_METADATA_DST", self.extended_error_dir())
            .env("RUSTC_BREAK_ON_ICE", "1");

        // Set RUSTC_WRAPPER to the bootstrap shim, which switches between beta and in-tree
        // sysroot depending on whether we're building build scripts.
        // NOTE: we intentionally use RUSTC_WRAPPER so that we can support clippy - RUSTC is not
        // respected by clippy-driver; RUSTC_WRAPPER happens earlier, before clippy runs.
        cargo.env("RUSTC_WRAPPER", self.bootstrap_out.join("rustc"));
        // NOTE: we also need to set RUSTC so cargo can run `rustc -vV`; apparently that ignores RUSTC_WRAPPER >:(
        cargo.env("RUSTC", self.bootstrap_out.join("rustc"));

        // Someone might have set some previous rustc wrapper (e.g.
        // sccache) before bootstrap overrode it. Respect that variable.
        if let Some(existing_wrapper) = env::var_os("RUSTC_WRAPPER") {
            cargo.env("RUSTC_WRAPPER_REAL", existing_wrapper);
        }

        // If this is for `miri-test`, prepare the sysroots.
        if cmd_kind == Kind::MiriTest {
            self.ensure(compile::Std::new(compiler, compiler.host));
            let host_sysroot = self.sysroot(compiler);
            let miri_sysroot = test::Miri::build_miri_sysroot(self, compiler, target);
            cargo.env("MIRI_SYSROOT", &miri_sysroot);
            cargo.env("MIRI_HOST_SYSROOT", &host_sysroot);
        }

        cargo.env(profile_var("STRIP"), self.config.rust_strip.to_string());

        if let Some(stack_protector) = &self.config.rust_stack_protector {
            rustflags.arg(&format!("-Zstack-protector={stack_protector}"));
        }

        if !matches!(cmd_kind, Kind::Build | Kind::Check | Kind::Clippy | Kind::Fix) && want_rustdoc
        {
            cargo.env("RUSTDOC_LIBDIR", self.rustc_libdir(compiler));
        }

        let debuginfo_level = match mode {
            Mode::Rustc | Mode::Codegen => self.config.rust_debuginfo_level_rustc,
            Mode::Std => self.config.rust_debuginfo_level_std,
            Mode::ToolBootstrap | Mode::ToolStd | Mode::ToolRustc => {
                self.config.rust_debuginfo_level_tools
            }
        };
        cargo.env(profile_var("DEBUG"), debuginfo_level.to_string());
        if let Some(opt_level) = &self.config.rust_optimize.get_opt_level() {
            cargo.env(profile_var("OPT_LEVEL"), opt_level);
        }
        cargo.env(
            profile_var("DEBUG_ASSERTIONS"),
            match mode {
                Mode::Std => self.config.std_debug_assertions,
                Mode::Rustc => self.config.rustc_debug_assertions,
                Mode::Codegen => self.config.rustc_debug_assertions,
                Mode::ToolBootstrap => self.config.tools_debug_assertions,
                Mode::ToolStd => self.config.tools_debug_assertions,
                Mode::ToolRustc => self.config.tools_debug_assertions,
            }
            .to_string(),
        );
        cargo.env(
            profile_var("OVERFLOW_CHECKS"),
            if mode == Mode::Std {
                self.config.rust_overflow_checks_std.to_string()
            } else {
                self.config.rust_overflow_checks.to_string()
            },
        );

        match self.config.split_debuginfo(target) {
            SplitDebuginfo::Packed => rustflags.arg("-Csplit-debuginfo=packed"),
            SplitDebuginfo::Unpacked => rustflags.arg("-Csplit-debuginfo=unpacked"),
            SplitDebuginfo::Off => rustflags.arg("-Csplit-debuginfo=off"),
        };

        if self.config.cmd.bless() {
            // Bless `expect!` tests.
            cargo.env("UPDATE_EXPECT", "1");
        }

        if !mode.is_tool() {
            cargo.env("RUSTC_FORCE_UNSTABLE", "1");
        }

        if let Some(x) = self.crt_static(target) {
            if x {
                rustflags.arg("-Ctarget-feature=+crt-static");
            } else {
                rustflags.arg("-Ctarget-feature=-crt-static");
            }
        }

        if let Some(x) = self.crt_static(compiler.host) {
            let sign = if x { "+" } else { "-" };
            hostflags.arg(format!("-Ctarget-feature={sign}crt-static"));
        }

        if let Some(map_to) = self.build.debuginfo_map_to(GitRepo::Rustc) {
            let map = format!("{}={}", self.build.src.display(), map_to);
            cargo.env("RUSTC_DEBUGINFO_MAP", map);

            // `rustc` needs to know the virtual `/rustc/$hash` we're mapping to,
            // in order to opportunistically reverse it later.
            cargo.env("CFG_VIRTUAL_RUST_SOURCE_BASE_DIR", map_to);
        }

        if self.config.rust_remap_debuginfo {
            let mut env_var = OsString::new();
            if let Some(vendor) = self.build.vendored_crates_path() {
                env_var.push(vendor);
                env_var.push("=/rust/deps");
            } else {
                let registry_src = t!(home::cargo_home()).join("registry").join("src");
                for entry in t!(std::fs::read_dir(registry_src)) {
                    if !env_var.is_empty() {
                        env_var.push("\t");
                    }
                    env_var.push(t!(entry).path());
                    env_var.push("=/rust/deps");
                }
            }
            cargo.env("RUSTC_CARGO_REGISTRY_SRC_TO_REMAP", env_var);
        }

        // Enable usage of unstable features
        cargo.env("RUSTC_BOOTSTRAP", "1");

        if self.config.dump_bootstrap_shims {
            prepare_behaviour_dump_dir(self.build);

            cargo
                .env("DUMP_BOOTSTRAP_SHIMS", self.build.out.join("bootstrap-shims-dump"))
                .env("BUILD_OUT", &self.build.out)
                .env("CARGO_HOME", t!(home::cargo_home()));
        };

        self.add_rust_test_threads(&mut cargo);

        // Almost all of the crates that we compile as part of the bootstrap may
        // have a build script, including the standard library. To compile a
        // build script, however, it itself needs a standard library! This
        // introduces a bit of a pickle when we're compiling the standard
        // library itself.
        //
        // To work around this we actually end up using the snapshot compiler
        // (stage0) for compiling build scripts of the standard library itself.
        // The stage0 compiler is guaranteed to have a libstd available for use.
        //
        // For other crates, however, we know that we've already got a standard
        // library up and running, so we can use the normal compiler to compile
        // build scripts in that situation.
        if mode == Mode::Std {
            cargo
                .env("RUSTC_SNAPSHOT", &self.initial_rustc)
                .env("RUSTC_SNAPSHOT_LIBDIR", self.rustc_snapshot_libdir());
        } else {
            cargo
                .env("RUSTC_SNAPSHOT", self.rustc(compiler))
                .env("RUSTC_SNAPSHOT_LIBDIR", self.rustc_libdir(compiler));
        }

        // Tools that use compiler libraries may inherit the `-lLLVM` link
        // requirement, but the `-L` library path is not propagated across
        // separate Cargo projects. We can add LLVM's library path to the
        // rustc args as a workaround.
        if mode == Mode::ToolRustc || mode == Mode::Codegen {
            if let Some(llvm_config) = self.llvm_config(target) {
                let llvm_libdir =
                    command(llvm_config).arg("--libdir").run_capture_stdout(self).stdout();
                if target.is_msvc() {
                    rustflags.arg(&format!("-Clink-arg=-LIBPATH:{llvm_libdir}"));
                } else {
                    rustflags.arg(&format!("-Clink-arg=-L{llvm_libdir}"));
                }
            }
        }

        // Compile everything except libraries and proc macros with the more
        // efficient initial-exec TLS model. This doesn't work with `dlopen`,
        // so we can't use it by default in general, but we can use it for tools
        // and our own internal libraries.
        //
        // Cygwin only supports emutls.
        if !mode.must_support_dlopen()
            && !target.triple.starts_with("powerpc-")
            && !target.triple.contains("cygwin")
        {
            cargo.env("RUSTC_TLS_MODEL_INITIAL_EXEC", "1");
        }

        // Ignore incremental modes except for stage0, since we're
        // not guaranteeing correctness across builds if the compiler
        // is changing under your feet.
        if self.config.incremental && compiler.stage == 0 {
            cargo.env("CARGO_INCREMENTAL", "1");
        } else {
            // Don't rely on any default setting for incr. comp. in Cargo
            cargo.env("CARGO_INCREMENTAL", "0");
        }

        if let Some(ref on_fail) = self.config.on_fail {
            cargo.env("RUSTC_ON_FAIL", on_fail);
        }

        if self.config.print_step_timings {
            cargo.env("RUSTC_PRINT_STEP_TIMINGS", "1");
        }

        if self.config.print_step_rusage {
            cargo.env("RUSTC_PRINT_STEP_RUSAGE", "1");
        }

        if self.config.backtrace_on_ice {
            cargo.env("RUSTC_BACKTRACE_ON_ICE", "1");
        }

        if self.is_verbose() {
            // This provides very useful logs especially when debugging build cache-related stuff.
            cargo.env("CARGO_LOG", "cargo::core::compiler::fingerprint=info");
        }

        cargo.env("RUSTC_VERBOSE", self.verbosity.to_string());

        // Downstream forks of the Rust compiler might want to use a custom libc to add support for
        // targets that are not yet available upstream. Adding a patch to replace libc with a
        // custom one would cause compilation errors though, because Cargo would interpret the
        // custom libc as part of the workspace, and apply the check-cfg lints on it.
        //
        // The libc build script emits check-cfg flags only when this environment variable is set,
        // so this line allows the use of custom libcs.
        cargo.env("LIBC_CHECK_CFG", "1");

        let mut lint_flags = Vec::new();

        // Lints for all in-tree code: compiler, rustdoc, cranelift, gcc,
        // clippy, rustfmt, rust-analyzer, etc.
        if source_type == SourceType::InTree {
            // When extending this list, add the new lints to the RUSTFLAGS of the
            // build_bootstrap function of src/bootstrap/bootstrap.py as well as
            // some code doesn't go through this `rustc` wrapper.
            lint_flags.push("-Wrust_2018_idioms");
            lint_flags.push("-Wunused_lifetimes");

            if self.config.deny_warnings {
                lint_flags.push("-Dwarnings");
                rustdocflags.arg("-Dwarnings");
            }

            rustdocflags.arg("-Wrustdoc::invalid_codeblock_attributes");
        }

        // Lints just for `compiler/` crates.
        if mode == Mode::Rustc {
            lint_flags.push("-Wrustc::internal");
            lint_flags.push("-Drustc::symbol_intern_string_literal");
            // FIXME(edition_2024): Change this to `-Wrust_2024_idioms` when all
            // of the individual lints are satisfied.
            lint_flags.push("-Wkeyword_idents_2024");
            lint_flags.push("-Wunreachable_pub");
            lint_flags.push("-Wunsafe_op_in_unsafe_fn");
            lint_flags.push("-Wunused_crate_dependencies");
        }

        // This does not use RUSTFLAGS for two reasons.
        // - Due to caching issues with Cargo. Clippy is treated as an "in
        //   tree" tool, but shares the same cache as other "submodule" tools.
        //   With these options set in RUSTFLAGS, that causes *every* shared
        //   dependency to be rebuilt. By injecting this into the rustc
        //   wrapper, this circumvents Cargo's fingerprint detection. This is
        //   fine because lint flags are always ignored in dependencies.
        //   Eventually this should be fixed via better support from Cargo.
        // - RUSTFLAGS is ignored for proc macro crates that are being built on
        //   the host (because `--target` is given). But we want the lint flags
        //   to be applied to proc macro crates.
        cargo.env("RUSTC_LINT_FLAGS", lint_flags.join(" "));

        if self.config.rust_frame_pointers {
            rustflags.arg("-Cforce-frame-pointers=true");
        }

        // If Control Flow Guard is enabled, pass the `control-flow-guard` flag to rustc
        // when compiling the standard library, since this might be linked into the final outputs
        // produced by rustc. Since this mitigation is only available on Windows, only enable it
        // for the standard library in case the compiler is run on a non-Windows platform.
        // This is not needed for stage 0 artifacts because these will only be used for building
        // the stage 1 compiler.
        if cfg!(windows)
            && mode == Mode::Std
            && self.config.control_flow_guard
            && compiler.stage >= 1
        {
            rustflags.arg("-Ccontrol-flow-guard");
        }

        // If EHCont Guard is enabled, pass the `-Zehcont-guard` flag to rustc when compiling the
        // standard library, since this might be linked into the final outputs produced by rustc.
        // Since this mitigation is only available on Windows, only enable it for the standard
        // library in case the compiler is run on a non-Windows platform.
        // This is not needed for stage 0 artifacts because these will only be used for building
        // the stage 1 compiler.
        if cfg!(windows) && mode == Mode::Std && self.config.ehcont_guard && compiler.stage >= 1 {
            rustflags.arg("-Zehcont-guard");
        }

        // For `cargo doc` invocations, make rustdoc print the Rust version into the docs
        // This replaces spaces with tabs because RUSTDOCFLAGS does not
        // support arguments with regular spaces. Hopefully someday Cargo will
        // have space support.
        let rust_version = self.rust_version().replace(' ', "\t");
        rustdocflags.arg("--crate-version").arg(&rust_version);

        // Environment variables *required* throughout the build
        //
        // FIXME: should update code to not require this env var

        // The host this new compiler will *run* on.
        cargo.env("CFG_COMPILER_HOST_TRIPLE", target.triple);
        // The host this new compiler is being *built* on.
        cargo.env("CFG_COMPILER_BUILD_TRIPLE", compiler.host.triple);

        // Set this for all builds to make sure doc builds also get it.
        cargo.env("CFG_RELEASE_CHANNEL", &self.config.channel);

        // This one's a bit tricky. As of the time of this writing the compiler
        // links to the `winapi` crate on crates.io. This crate provides raw
        // bindings to Windows system functions, sort of like libc does for
        // Unix. This crate also, however, provides "import libraries" for the
        // MinGW targets. There's an import library per dll in the windows
        // distribution which is what's linked to. These custom import libraries
        // are used because the winapi crate can reference Windows functions not
        // present in the MinGW import libraries.
        //
        // For example MinGW may ship libdbghelp.a, but it may not have
        // references to all the functions in the dbghelp dll. Instead the
        // custom import library for dbghelp in the winapi crates has all this
        // information.
        //
        // Unfortunately for us though the import libraries are linked by
        // default via `-ldylib=winapi_foo`. That is, they're linked with the
        // `dylib` type with a `winapi_` prefix (so the winapi ones don't
        // conflict with the system MinGW ones). This consequently means that
        // the binaries we ship of things like rustc_codegen_llvm (aka the rustc_codegen_llvm
        // DLL) when linked against *again*, for example with procedural macros
        // or plugins, will trigger the propagation logic of `-ldylib`, passing
        // `-lwinapi_foo` to the linker again. This isn't actually available in
        // our distribution, however, so the link fails.
        //
        // To solve this problem we tell winapi to not use its bundled import
        // libraries. This means that it will link to the system MinGW import
        // libraries by default, and the `-ldylib=foo` directives will still get
        // passed to the final linker, but they'll look like `-lfoo` which can
        // be resolved because MinGW has the import library. The downside is we
        // don't get newer functions from Windows, but we don't use any of them
        // anyway.
        if !mode.is_tool() {
            cargo.env("WINAPI_NO_BUNDLED_LIBRARIES", "1");
        }

        for _ in 0..self.verbosity {
            cargo.arg("-v");
        }

        match (mode, self.config.rust_codegen_units_std, self.config.rust_codegen_units) {
            (Mode::Std, Some(n), _) | (_, _, Some(n)) => {
                cargo.env(profile_var("CODEGEN_UNITS"), n.to_string());
            }
            _ => {
                // Don't set anything
            }
        }

        if self.config.locked_deps {
            cargo.arg("--locked");
        }
        if self.config.vendor || self.is_sudo {
            cargo.arg("--frozen");
        }

        // Try to use a sysroot-relative bindir, in case it was configured absolutely.
        cargo.env("RUSTC_INSTALL_BINDIR", self.config.bindir_relative());

        cargo.force_coloring_in_ci();

        // When we build Rust dylibs they're all intended for intermediate
        // usage, so make sure we pass the -Cprefer-dynamic flag instead of
        // linking all deps statically into the dylib.
        if matches!(mode, Mode::Std) {
            rustflags.arg("-Cprefer-dynamic");
        }
        if matches!(mode, Mode::Rustc) && !self.link_std_into_rustc_driver(target) {
            rustflags.arg("-Cprefer-dynamic");
        }

        cargo.env(
            "RUSTC_LINK_STD_INTO_RUSTC_DRIVER",
            if self.link_std_into_rustc_driver(target) { "1" } else { "0" },
        );

        // When building incrementally we default to a lower ThinLTO import limit
        // (unless explicitly specified otherwise). This will produce a somewhat
        // slower code but give way better compile times.
        {
            let limit = match self.config.rust_thin_lto_import_instr_limit {
                Some(limit) => Some(limit),
                None if self.config.incremental => Some(10),
                _ => None,
            };

            if let Some(limit) = limit {
                if stage == 0
                    || self.config.default_codegen_backend(target).unwrap_or_default() == "llvm"
                {
                    rustflags.arg(&format!("-Cllvm-args=-import-instr-limit={limit}"));
                }
            }
        }

        if matches!(mode, Mode::Std) {
            if let Some(mir_opt_level) = self.config.rust_validate_mir_opts {
                rustflags.arg("-Zvalidate-mir");
                rustflags.arg(&format!("-Zmir-opt-level={mir_opt_level}"));
            }
            if self.config.rust_randomize_layout {
                rustflags.arg("--cfg=randomized_layouts");
            }
            // Always enable inlining MIR when building the standard library.
            // Without this flag, MIR inlining is disabled when incremental compilation is enabled.
            // That causes some mir-opt tests which inline functions from the standard library to
            // break when incremental compilation is enabled. So this overrides the "no inlining
            // during incremental builds" heuristic for the standard library.
            rustflags.arg("-Zinline-mir");

            // Similarly, we need to keep debug info for functions inlined into other std functions,
            // even if we're not going to output debuginfo for the crate we're currently building,
            // so that it'll be available when downstream consumers of std try to use it.
            rustflags.arg("-Zinline-mir-preserve-debug");

            rustflags.arg("-Zmir_strip_debuginfo=locals-in-tiny-functions");
        }

        let release_build = self.config.rust_optimize.is_release() &&
            // cargo bench/install do not accept `--release` and miri doesn't want it
            !matches!(cmd_kind, Kind::Bench | Kind::Install | Kind::Miri | Kind::MiriSetup | Kind::MiriTest);

        Cargo {
            command: cargo,
            args: vec![],
            compiler,
            target,
            rustflags,
            rustdocflags,
            hostflags,
            allow_features,
            release_build,
        }
    }
}

pub fn cargo_profile_var(name: &str, config: &Config) -> String {
    let profile = if config.rust_optimize.is_release() { "RELEASE" } else { "DEV" };
    format!("CARGO_PROFILE_{profile}_{name}")
}
