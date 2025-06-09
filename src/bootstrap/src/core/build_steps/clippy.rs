//! Implementation of running clippy on the compiler, standard library and various tools.

use super::compile::{run_cargo, rustc_cargo, std_cargo};
use super::tool::{SourceType, prepare_tool_cargo};
use super::{check, compile};
use crate::builder::{Builder, ShouldRun};
use crate::core::build_steps::compile::std_crates_for_run_make;
use crate::core::builder;
use crate::core::builder::{Alias, Kind, RunConfig, Step, crate_description};
use crate::utils::build_stamp::{self, BuildStamp};
use crate::{Mode, Subcommand, TargetSelection};

/// Disable the most spammy clippy lints
const IGNORED_RULES_FOR_STD_AND_RUSTC: &[&str] = &[
    "many_single_char_names", // there are a lot in stdarch
    "collapsible_if",
    "type_complexity",
    "missing_safety_doc", // almost 3K warnings
    "too_many_arguments",
    "needless_lifetimes", // people want to keep the lifetimes
    "wrong_self_convention",
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
    pub target: TargetSelection,
    config: LintConfig,
    /// Whether to lint only a subset of crates.
    crates: Vec<String>,
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
        run.builder.ensure(Std { target: run.target, config, crates });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.require_submodule("library/stdarch", None);

        let target = self.target;
        let compiler = builder.compiler(builder.top_stage, builder.config.build);

        let mut cargo = builder::Cargo::new(
            builder,
            compiler,
            Mode::Std,
            SourceType::InTree,
            target,
            Kind::Clippy,
        );

        std_cargo(builder, target, compiler.stage, &mut cargo);

        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let _guard =
            builder.msg_clippy(format_args!("library{}", crate_description(&self.crates)), target);

        run_cargo(
            builder,
            cargo,
            lint_args(builder, &self.config, IGNORED_RULES_FOR_STD_AND_RUSTC),
            &build_stamp::libstd_stamp(builder, compiler, target),
            vec![],
            true,
            false,
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rustc {
    pub target: TargetSelection,
    config: LintConfig,
    /// Whether to lint only a subset of crates.
    crates: Vec<String>,
}

impl Step for Rustc {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("rustc-main").path("compiler")
    }

    fn make_run(run: RunConfig<'_>) {
        let crates = run.make_run_crates(Alias::Compiler);
        let config = LintConfig::new(run.builder);
        run.builder.ensure(Rustc { target: run.target, config, crates });
    }

    /// Lints the compiler.
    ///
    /// This will lint the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture.
    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(builder.top_stage, builder.config.build);
        let target = self.target;

        if !builder.download_rustc() {
            if compiler.stage != 0 {
                // If we're not in stage 0, then we won't have a std from the beta
                // compiler around. That means we need to make sure there's one in
                // the sysroot for the compiler to find. Otherwise, we're going to
                // fail when building crates that need to generate code (e.g., build
                // scripts and their dependencies).
                builder.ensure(compile::Std::new(compiler, compiler.host));
                builder.ensure(compile::Std::new(compiler, target));
            } else {
                builder.ensure(check::Std::new(target).build_kind(Some(Kind::Check)));
            }
        }

        let mut cargo = builder::Cargo::new(
            builder,
            compiler,
            Mode::Rustc,
            SourceType::InTree,
            target,
            Kind::Clippy,
        );

        rustc_cargo(builder, &mut cargo, target, &compiler, &self.crates);

        // Explicitly pass -p for all compiler crates -- this will force cargo
        // to also lint the tests/benches/examples for these crates, rather
        // than just the leaf crate.
        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let _guard =
            builder.msg_clippy(format_args!("compiler{}", crate_description(&self.crates)), target);

        run_cargo(
            builder,
            cargo,
            lint_args(builder, &self.config, IGNORED_RULES_FOR_STD_AND_RUSTC),
            &build_stamp::librustc_stamp(builder, compiler, target),
            vec![],
            true,
            false,
        );
    }
}

macro_rules! lint_any {
    ($(
        $name:ident, $path:expr, $readable_name:expr
        $(,lint_by_default = $lint_by_default:expr)*
        ;
    )+) => {
        $(

        #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            pub target: TargetSelection,
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
                    target: run.target,
                    config,
                });
            }

            fn run(self, builder: &Builder<'_>) -> Self::Output {
                let compiler = builder.compiler(builder.top_stage, builder.config.build);
                let target = self.target;

                if !builder.download_rustc() {
                    builder.ensure(check::Rustc::new(target, builder).build_kind(Some(Kind::Check)));
                };

                let cargo = prepare_tool_cargo(
                    builder,
                    compiler,
                    Mode::ToolRustc,
                    target,
                    Kind::Clippy,
                    $path,
                    SourceType::InTree,
                    &[],
                );

                let _guard = builder.msg_tool(
                    Kind::Clippy,
                    Mode::ToolRustc,
                    $readable_name,
                    compiler.stage,
                    &compiler.host,
                    &target,
                );

                let stringified_name = stringify!($name).to_lowercase();
                let stamp = BuildStamp::new(&builder.cargo_out(compiler, Mode::ToolRustc, target))
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
        }
        )+
    }
}

lint_any!(
    Bootstrap, "src/bootstrap", "bootstrap";
    BuildHelper, "src/build_helper", "build_helper";
    BuildManifest, "src/tools/build-manifest", "build-manifest";
    CargoMiri, "src/tools/miri/cargo-miri", "cargo-miri";
    Clippy, "src/tools/clippy", "clippy";
    CollectLicenseMetadata, "src/tools/collect-license-metadata", "collect-license-metadata";
    CodegenGcc, "compiler/rustc_codegen_gcc", "rustc-codegen-gcc";
    Compiletest, "src/tools/compiletest", "compiletest";
    CoverageDump, "src/tools/coverage-dump", "coverage-dump";
    Jsondocck, "src/tools/jsondocck", "jsondocck";
    Jsondoclint, "src/tools/jsondoclint", "jsondoclint";
    LintDocs, "src/tools/lint-docs", "lint-docs";
    LlvmBitcodeLinker, "src/tools/llvm-bitcode-linker", "llvm-bitcode-linker";
    Miri, "src/tools/miri", "miri";
    MiroptTestTools, "src/tools/miropt-test-tools", "miropt-test-tools";
    OptDist, "src/tools/opt-dist", "opt-dist";
    RemoteTestClient, "src/tools/remote-test-client", "remote-test-client";
    RemoteTestServer, "src/tools/remote-test-server", "remote-test-server";
    RustAnalyzer, "src/tools/rust-analyzer", "rust-analyzer";
    Rustdoc, "src/librustdoc", "clippy";
    Rustfmt, "src/tools/rustfmt", "rustfmt";
    RustInstaller, "src/tools/rust-installer", "rust-installer";
    Tidy, "src/tools/tidy", "tidy";
    TestFloatParse, "src/tools/test-float-parse", "test-float-parse";
);

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
        builder.ensure(Bootstrap {
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
        builder.ensure(Std {
            target: self.target,
            config: self.config.merge(&library_clippy_cfg),
            crates: vec![],
        });

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
        builder.ensure(Rustc {
            target: self.target,
            config: self.config.merge(&compiler_clippy_cfg),
            crates: vec![],
        });

        let rustc_codegen_gcc = LintConfig {
            allow: vec![],
            warn: vec![],
            deny: vec!["warnings".into()],
            forbid: vec![],
        };
        builder.ensure(CodegenGcc {
            target: self.target,
            config: self.config.merge(&rustc_codegen_gcc),
        });
    }
}
