//! Implementation of running clippy on the compiler, standard library and various tools.
//!
//! This serves a double purpose:
//! - The first is to run Clippy itself on in-tree code, in order to test and dogfood it.
//! - The second is to actually lint the in-tree codebase on CI, with a hard-coded set of rules,
//!   which is performed by the `x clippy ci` command.
//!
//! In order to prepare a build compiler for running clippy, use the
//! [prepare_compiler_for_check] function. That prepares a
//! compiler and a standard library
//! for running Clippy. The second part (actually building Clippy) is performed inside
//! [Builder::cargo_clippy_cmd]. It would be nice if this was more explicit, and we actually had
//! to pass a prebuilt Clippy from the outside when running `cargo clippy`, but that would be
//! (as usual) a massive undertaking/refactoring.

use build_helper::exit;

use super::compile::{run_cargo, rustc_cargo, std_cargo};
use super::tool::{SourceType, prepare_tool_cargo};
use crate::builder::{Builder, ShouldRun};
use crate::core::build_steps::check::{CompilerForCheck, prepare_compiler_for_check};
use crate::core::build_steps::compile::std_crates_for_run_make;
use crate::core::builder;
use crate::core::builder::{Alias, Kind, RunConfig, Step, StepMetadata, crate_description};
use crate::utils::build_stamp::{self, BuildStamp};
use crate::{Compiler, Mode, Subcommand, TargetSelection};

/// Disable the most spammy clippy lints
const IGNORED_RULES_FOR_STD_AND_RUSTC: &[&str] = &[
    "many_single_char_names", // there are a lot in stdarch
    "collapsible_if",
    "type_complexity",
    "missing_safety_doc", // almost 3K warnings
    "too_many_arguments",
    "needless_lifetimes", // people want to keep the lifetimes
    "wrong_self_convention",
    "approx_constant", // libcore is what defines those
];

fn lint_args(builder: &Builder<'_>, config: &LintConfig, ignored_rules: &[&str]) -> Vec<String> {
    fn strings<'a>(arr: &'a [&str]) -> impl Iterator<Item = String> + 'a {
        arr.iter().copied().map(String::from)
    }

    let Subcommand::Clippy { fix, allow_dirty, allow_staged, .. } = &builder.config.cmd else {
        unreachable!("clippy::lint_args can only be called from `clippy` subcommands.");
    };

    let mut args = vec![];
    if *fix {
        #[rustfmt::skip]
            args.extend(strings(&[
                "--fix", "-Zunstable-options",
                // FIXME: currently, `--fix` gives an error while checking tests for libtest,
                // possibly because libtest is not yet built in the sysroot.
                // As a workaround, avoid checking tests and benches when passed --fix.
                "--lib", "--bins", "--examples",
            ]));

        if *allow_dirty {
            args.push("--allow-dirty".to_owned());
        }

        if *allow_staged {
            args.push("--allow-staged".to_owned());
        }
    }

    args.extend(strings(&["--"]));

    if config.deny.is_empty() && config.forbid.is_empty() {
        args.extend(strings(&["--cap-lints", "warn"]));
    }

    let all_args = std::env::args().collect::<Vec<_>>();
    args.extend(get_clippy_rules_in_order(&all_args, config));

    args.extend(ignored_rules.iter().map(|lint| format!("-Aclippy::{lint}")));
    args.extend(builder.config.free_args.clone());
    args
}

/// We need to keep the order of the given clippy lint rules before passing them.
/// Since clap doesn't offer any useful interface for this purpose out of the box,
/// we have to handle it manually.
pub fn get_clippy_rules_in_order(all_args: &[String], config: &LintConfig) -> Vec<String> {
    let mut result = vec![];

    for (prefix, item) in
        [("-A", &config.allow), ("-D", &config.deny), ("-W", &config.warn), ("-F", &config.forbid)]
    {
        item.iter().for_each(|v| {
            let rule = format!("{prefix}{v}");
            // Arguments added by bootstrap in LintConfig won't show up in the all_args list, so
            // put them at the end of the command line.
            let position = all_args.iter().position(|t| t == &rule || t == v).unwrap_or(usize::MAX);
            result.push((position, rule));
        });
    }

    result.sort_by_key(|&(position, _)| position);
    result.into_iter().map(|v| v.1).collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LintConfig {
    pub allow: Vec<String>,
    pub warn: Vec<String>,
    pub deny: Vec<String>,
    pub forbid: Vec<String>,
}

impl LintConfig {
    fn new(builder: &Builder<'_>) -> Self {
        match builder.config.cmd.clone() {
            Subcommand::Clippy { allow, deny, warn, forbid, .. } => {
                Self { allow, warn, deny, forbid }
            }
            _ => unreachable!("LintConfig can only be called from `clippy` subcommands."),
        }
    }

    fn merge(&self, other: &Self) -> Self {
        let merged = |self_attr: &[String], other_attr: &[String]| -> Vec<String> {
            self_attr.iter().cloned().chain(other_attr.iter().cloned()).collect()
        };
        // This is written this way to ensure we get a compiler error if we add a new field.
        Self {
            allow: merged(&self.allow, &other.allow),
            warn: merged(&self.warn, &other.warn),
            deny: merged(&self.deny, &other.deny),
            forbid: merged(&self.forbid, &other.forbid),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Std {
    build_compiler: Compiler,
    target: TargetSelection,
    config: LintConfig,
    /// Whether to lint only a subset of crates.
    crates: Vec<String>,
}

impl Std {
    fn new(
        builder: &Builder<'_>,
        target: TargetSelection,
        config: LintConfig,
        crates: Vec<String>,
    ) -> Self {
        Self {
            build_compiler: builder.compiler(builder.top_stage, builder.host_target),
            target,
            config,
            crates,
        }
    }

    fn from_build_compiler(
        build_compiler: Compiler,
        target: TargetSelection,
        config: LintConfig,
        crates: Vec<String>,
    ) -> Self {
        Self { build_compiler, target, config, crates }
    }
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("sysroot").path("library")
    }

    fn make_run(run: RunConfig<'_>) {
        let crates = std_crates_for_run_make(&run);
        let config = LintConfig::new(run.builder);
        run.builder.ensure(Std::new(run.builder, run.target, config, crates));
    }

    fn run(self, builder: &Builder<'_>) {
        let target = self.target;
        let build_compiler = self.build_compiler;

        let mut cargo = builder::Cargo::new(
            builder,
            build_compiler,
            Mode::Std,
            SourceType::InTree,
            target,
            Kind::Clippy,
        );

        std_cargo(builder, target, &mut cargo);

        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let _guard = builder.msg(
            Kind::Clippy,
            format_args!("library{}", crate_description(&self.crates)),
            Mode::Std,
            build_compiler,
            target,
        );

        run_cargo(
            builder,
            cargo,
            lint_args(builder, &self.config, IGNORED_RULES_FOR_STD_AND_RUSTC),
            &build_stamp::libstd_stamp(builder, build_compiler, target),
            vec![],
            true,
            false,
        );
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::clippy("std", self.target).built_by(self.build_compiler))
    }
}

/// Lints the compiler.
///
/// This will build Clippy with the `build_compiler` and use it to lint
/// in-tree rustc.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rustc {
    build_compiler: CompilerForCheck,
    target: TargetSelection,
    config: LintConfig,
    /// Whether to lint only a subset of crates.
    crates: Vec<String>,
}

impl Rustc {
    fn new(
        builder: &Builder<'_>,
        target: TargetSelection,
        config: LintConfig,
        crates: Vec<String>,
    ) -> Self {
        Self {
            build_compiler: prepare_compiler_for_check(builder, target, Mode::Rustc),
            target,
            config,
            crates,
        }
    }
}

impl Step for Rustc {
    type Output = ();
    const IS_HOST: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("rustc-main").path("compiler")
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let crates = run.make_run_crates(Alias::Compiler);
        let config = LintConfig::new(run.builder);
        run.builder.ensure(Rustc::new(builder, run.target, config, crates));
    }

    fn run(self, builder: &Builder<'_>) {
        let build_compiler = self.build_compiler.build_compiler();
        let target = self.target;

        let mut cargo = builder::Cargo::new(
            builder,
            build_compiler,
            Mode::Rustc,
            SourceType::InTree,
            target,
            Kind::Clippy,
        );

        rustc_cargo(builder, &mut cargo, target, &build_compiler, &self.crates);
        self.build_compiler.configure_cargo(&mut cargo);

        // Explicitly pass -p for all compiler crates -- this will force cargo
        // to also lint the tests/benches/examples for these crates, rather
        // than just the leaf crate.
        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let _guard = builder.msg(
            Kind::Clippy,
            format_args!("compiler{}", crate_description(&self.crates)),
            Mode::Rustc,
            build_compiler,
            target,
        );

        run_cargo(
            builder,
            cargo,
            lint_args(builder, &self.config, IGNORED_RULES_FOR_STD_AND_RUSTC),
            &build_stamp::librustc_stamp(builder, build_compiler, target),
            vec![],
            true,
            false,
        );
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(
            StepMetadata::clippy("rustc", self.target)
                .built_by(self.build_compiler.build_compiler()),
        )
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CodegenGcc {
    build_compiler: CompilerForCheck,
    target: TargetSelection,
    config: LintConfig,
}

impl CodegenGcc {
    fn new(builder: &Builder<'_>, target: TargetSelection, config: LintConfig) -> Self {
        Self {
            build_compiler: prepare_compiler_for_check(builder, target, Mode::Codegen),
            target,
            config,
        }
    }
}

impl Step for CodegenGcc {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("rustc_codegen_gcc")
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let config = LintConfig::new(builder);
        builder.ensure(CodegenGcc::new(builder, run.target, config));
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let build_compiler = self.build_compiler.build_compiler();
        let target = self.target;

        let mut cargo = prepare_tool_cargo(
            builder,
            build_compiler,
            Mode::Codegen,
            target,
            Kind::Clippy,
            "compiler/rustc_codegen_gcc",
            SourceType::InTree,
            &[],
        );
        self.build_compiler.configure_cargo(&mut cargo);

        let _guard = builder.msg(
            Kind::Clippy,
            "rustc_codegen_gcc",
            Mode::ToolRustcPrivate,
            build_compiler,
            target,
        );

        let stamp = BuildStamp::new(&builder.cargo_out(build_compiler, Mode::Codegen, target))
            .with_prefix("rustc_codegen_gcc-check");

        run_cargo(
            builder,
            cargo,
            lint_args(builder, &self.config, &[]),
            &stamp,
            vec![],
            true,
            false,
        );
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(
            StepMetadata::clippy("rustc_codegen_gcc", self.target)
                .built_by(self.build_compiler.build_compiler()),
        )
    }
}

macro_rules! lint_any {
    ($(
        $name:ident,
        $path:expr,
        $readable_name:expr,
        $mode:expr
        $(,lint_by_default = $lint_by_default:expr)*
        ;
    )+) => {
        $(

        #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            build_compiler: CompilerForCheck,
            target: TargetSelection,
            config: LintConfig,
        }

        impl Step for $name {
            type Output = ();
            const DEFAULT: bool = if false $(|| $lint_by_default)* { true } else { false };

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.path($path)
            }

            fn make_run(run: RunConfig<'_>) {
                let config = LintConfig::new(run.builder);
                run.builder.ensure($name {
                    build_compiler: prepare_compiler_for_check(run.builder, run.target, $mode),
                    target: run.target,
                    config,
                });
            }

            fn run(self, builder: &Builder<'_>) -> Self::Output {
                let build_compiler = self.build_compiler.build_compiler();
                let target = self.target;
                let mut cargo = prepare_tool_cargo(
                    builder,
                    build_compiler,
                    $mode,
                    target,
                    Kind::Clippy,
                    $path,
                    SourceType::InTree,
                    &[],
                );
                self.build_compiler.configure_cargo(&mut cargo);

                let _guard = builder.msg(
                    Kind::Clippy,
                    $readable_name,
                    $mode,
                    build_compiler,
                    target,
                );

                let stringified_name = stringify!($name).to_lowercase();
                let stamp = BuildStamp::new(&builder.cargo_out(build_compiler, $mode, target))
                    .with_prefix(&format!("{}-check", stringified_name));

                run_cargo(
                    builder,
                    cargo,
                    lint_args(builder, &self.config, &[]),
                    &stamp,
                    vec![],
                    true,
                    false,
                );
            }

            fn metadata(&self) -> Option<StepMetadata> {
                Some(StepMetadata::clippy($readable_name, self.target).built_by(self.build_compiler.build_compiler()))
            }
        }
        )+
    }
}

// Note: we use ToolTarget instead of ToolBootstrap here, to allow linting in-tree host tools
// using the in-tree Clippy. Because Mode::ToolBootstrap would always use stage 0 rustc/Clippy.
lint_any!(
    Bootstrap, "src/bootstrap", "bootstrap", Mode::ToolTarget;
    BuildHelper, "src/build_helper", "build_helper", Mode::ToolTarget;
    BuildManifest, "src/tools/build-manifest", "build-manifest", Mode::ToolTarget;
    CargoMiri, "src/tools/miri/cargo-miri", "cargo-miri", Mode::ToolRustcPrivate;
    Clippy, "src/tools/clippy", "clippy", Mode::ToolRustcPrivate;
    CollectLicenseMetadata, "src/tools/collect-license-metadata", "collect-license-metadata", Mode::ToolTarget;
    Compiletest, "src/tools/compiletest", "compiletest", Mode::ToolTarget;
    CoverageDump, "src/tools/coverage-dump", "coverage-dump", Mode::ToolTarget;
    Jsondocck, "src/tools/jsondocck", "jsondocck", Mode::ToolTarget;
    Jsondoclint, "src/tools/jsondoclint", "jsondoclint", Mode::ToolTarget;
    LintDocs, "src/tools/lint-docs", "lint-docs", Mode::ToolTarget;
    LlvmBitcodeLinker, "src/tools/llvm-bitcode-linker", "llvm-bitcode-linker", Mode::ToolTarget;
    Miri, "src/tools/miri", "miri", Mode::ToolRustcPrivate;
    MiroptTestTools, "src/tools/miropt-test-tools", "miropt-test-tools", Mode::ToolTarget;
    OptDist, "src/tools/opt-dist", "opt-dist", Mode::ToolTarget;
    RemoteTestClient, "src/tools/remote-test-client", "remote-test-client", Mode::ToolTarget;
    RemoteTestServer, "src/tools/remote-test-server", "remote-test-server", Mode::ToolTarget;
    RustAnalyzer, "src/tools/rust-analyzer", "rust-analyzer", Mode::ToolRustcPrivate;
    Rustdoc, "src/librustdoc", "clippy", Mode::ToolRustcPrivate;
    Rustfmt, "src/tools/rustfmt", "rustfmt", Mode::ToolRustcPrivate;
    RustInstaller, "src/tools/rust-installer", "rust-installer", Mode::ToolTarget;
    Tidy, "src/tools/tidy", "tidy", Mode::ToolTarget;
    TestFloatParse, "src/tools/test-float-parse", "test-float-parse", Mode::ToolStd;
);

/// Runs Clippy on in-tree sources of selected projects using in-tree CLippy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CI {
    target: TargetSelection,
    config: LintConfig,
}

impl Step for CI {
    type Output = ();
    const DEFAULT: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("ci")
    }

    fn make_run(run: RunConfig<'_>) {
        let config = LintConfig::new(run.builder);
        run.builder.ensure(CI { target: run.target, config });
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        if builder.top_stage != 2 {
            eprintln!("ERROR: `x clippy ci` should always be executed with --stage 2");
            exit!(1);
        }

        // We want to check in-tree source using in-tree clippy. However, if we naively did
        // a stage 2 `x clippy ci`, it would *build* a stage 2 rustc, in order to lint stage 2
        // std, which is wasteful.
        // So we want to lint stage 2 [bootstrap/rustc/...], but only stage 1 std rustc_codegen_gcc.
        // We thus construct the compilers in this step manually, to optimize the number of
        // steps that get built.

        builder.ensure(Bootstrap {
            // This will be the stage 1 compiler
            build_compiler: prepare_compiler_for_check(builder, self.target, Mode::ToolTarget),
            target: self.target,
            config: self.config.merge(&LintConfig {
                allow: vec![],
                warn: vec![],
                deny: vec!["warnings".into()],
                forbid: vec![],
            }),
        });

        let library_clippy_cfg = LintConfig {
            allow: vec!["clippy::all".into()],
            warn: vec![],
            deny: vec![
                "clippy::correctness".into(),
                "clippy::char_lit_as_u8".into(),
                "clippy::four_forward_slashes".into(),
                "clippy::needless_bool".into(),
                "clippy::needless_bool_assign".into(),
                "clippy::non_minimal_cfg".into(),
                "clippy::print_literal".into(),
                "clippy::same_item_push".into(),
                "clippy::single_char_add_str".into(),
                "clippy::to_string_in_format_args".into(),
            ],
            forbid: vec![],
        };
        builder.ensure(Std::from_build_compiler(
            // This will be the stage 1 compiler, to avoid building rustc stage 2 just to lint std
            builder.compiler(1, self.target),
            self.target,
            self.config.merge(&library_clippy_cfg),
            vec![],
        ));

        let compiler_clippy_cfg = LintConfig {
            allow: vec!["clippy::all".into()],
            warn: vec![],
            deny: vec![
                "clippy::correctness".into(),
                "clippy::char_lit_as_u8".into(),
                "clippy::clone_on_ref_ptr".into(),
                "clippy::format_in_format_args".into(),
                "clippy::four_forward_slashes".into(),
                "clippy::needless_bool".into(),
                "clippy::needless_bool_assign".into(),
                "clippy::non_minimal_cfg".into(),
                "clippy::print_literal".into(),
                "clippy::same_item_push".into(),
                "clippy::single_char_add_str".into(),
                "clippy::to_string_in_format_args".into(),
            ],
            forbid: vec![],
        };
        // This will lint stage 2 rustc using stage 1 Clippy
        builder.ensure(Rustc::new(
            builder,
            self.target,
            self.config.merge(&compiler_clippy_cfg),
            vec![],
        ));

        let rustc_codegen_gcc = LintConfig {
            allow: vec![],
            warn: vec![],
            deny: vec!["warnings".into()],
            forbid: vec![],
        };
        // This will check stage 2 rustc
        builder.ensure(CodegenGcc::new(
            builder,
            self.target,
            self.config.merge(&rustc_codegen_gcc),
        ));
    }
}
