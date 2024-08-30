//! Implementation of running clippy on the compiler, standard library and various tools.

use super::compile::{librustc_stamp, libstd_stamp, run_cargo, rustc_cargo, std_cargo};
use super::tool::{prepare_tool_cargo, SourceType};
use super::{check, compile};
use crate::builder::{Builder, ShouldRun};
use crate::core::build_steps::compile::std_crates_for_run_make;
use crate::core::builder;
use crate::core::builder::{crate_description, Alias, Kind, RunConfig, Step};
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

fn lint_args(builder: &Builder<'_>, ignored_rules: &[&str]) -> Vec<String> {
    fn strings<'a>(arr: &'a [&str]) -> impl Iterator<Item = String> + 'a {
        arr.iter().copied().map(String::from)
    }

    let Subcommand::Clippy { fix, allow_dirty, allow_staged, allow, deny, warn, forbid } =
        &builder.config.cmd
    else {
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

    if deny.is_empty() && forbid.is_empty() {
        args.extend(strings(&["--cap-lints", "warn"]));
    }

    let all_args = std::env::args().collect::<Vec<_>>();
    args.extend(get_clippy_rules_in_order(&all_args, allow, deny, warn, forbid));

    args.extend(ignored_rules.iter().map(|lint| format!("-Aclippy::{}", lint)));
    args.extend(builder.config.free_args.clone());
    args
}

/// We need to keep the order of the given clippy lint rules before passing them.
/// Since clap doesn't offer any useful interface for this purpose out of the box,
/// we have to handle it manually.
pub(crate) fn get_clippy_rules_in_order(
    all_args: &[String],
    allow_rules: &[String],
    deny_rules: &[String],
    warn_rules: &[String],
    forbid_rules: &[String],
) -> Vec<String> {
    let mut result = vec![];

    for (prefix, item) in
        [("-A", allow_rules), ("-D", deny_rules), ("-W", warn_rules), ("-F", forbid_rules)]
    {
        item.iter().for_each(|v| {
            let rule = format!("{prefix}{v}");
            let position = all_args.iter().position(|t| t == &rule).unwrap();
            result.push((position, rule));
        });
    }

    result.sort_by_key(|&(position, _)| position);
    result.into_iter().map(|v| v.1).collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Std {
    pub target: TargetSelection,
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
        run.builder.ensure(Std { target: run.target, crates });
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
            lint_args(builder, IGNORED_RULES_FOR_STD_AND_RUSTC),
            &libstd_stamp(builder, compiler, target),
            vec![],
            true,
            false,
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rustc {
    pub target: TargetSelection,
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
        run.builder.ensure(Rustc { target: run.target, crates });
    }

    /// Lints the compiler.
    ///
    /// This will lint the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture.
    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(builder.top_stage, builder.config.build);
        let target = self.target;

        if compiler.stage != 0 {
            // If we're not in stage 0, then we won't have a std from the beta
            // compiler around. That means we need to make sure there's one in
            // the sysroot for the compiler to find. Otherwise, we're going to
            // fail when building crates that need to generate code (e.g., build
            // scripts and their dependencies).
            builder.ensure(compile::Std::new(compiler, compiler.host));
            builder.ensure(compile::Std::new(compiler, target));
        } else {
            builder.ensure(check::Std::new_with_build_kind(target, Some(Kind::Check)));
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
            lint_args(builder, IGNORED_RULES_FOR_STD_AND_RUSTC),
            &librustc_stamp(builder, compiler, target),
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
        }

        impl Step for $name {
            type Output = ();
            const DEFAULT: bool = if false $(|| $lint_by_default)* { true } else { false };

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.path($path)
            }

            fn make_run(run: RunConfig<'_>) {
                run.builder.ensure($name {
                    target: run.target,
                });
            }

            fn run(self, builder: &Builder<'_>) -> Self::Output {
                let compiler = builder.compiler(builder.top_stage, builder.config.build);
                let target = self.target;

                builder.ensure(check::Rustc::new_with_build_kind(target, builder, Some(Kind::Check)));

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

                let stamp = builder
                    .cargo_out(compiler, Mode::ToolRustc, target)
                    .join(format!(".{}-check.stamp", stringify!($name).to_lowercase()));

                run_cargo(
                    builder,
                    cargo,
                    lint_args(builder, &[]),
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
    BuildHelper, "src/tools/build_helper", "build_helper";
    BuildManifest, "src/tools/build-manifest", "build-manifest";
    CargoMiri, "src/tools/miri/cargo-miri", "cargo-miri";
    Clippy, "src/tools/clippy", "clippy";
    CollectLicenseMetadata, "src/tools/collect-license-metadata", "collect-license-metadata";
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
    Rls, "src/tools/rls", "rls";
    RustAnalyzer, "src/tools/rust-analyzer", "rust-analyzer";
    Rustdoc, "src/librustdoc", "clippy";
    Rustfmt, "src/tools/rustfmt", "rustfmt";
    RustInstaller, "src/tools/rust-installer", "rust-installer";
    Tidy, "src/tools/tidy", "tidy";
    TestFloatParse, "src/etc/test-float-parse", "test-float-parse";
);
