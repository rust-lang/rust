use std::path::Path;

use crate::builder::Builder;
use crate::builder::ShouldRun;
use crate::core::builder;
use crate::core::builder::crate_description;
use crate::core::builder::Alias;
use crate::core::builder::RunConfig;
use crate::core::builder::Step;
use crate::Mode;
use crate::Subcommand;
use crate::TargetSelection;

use super::check;
use super::compile;
use super::compile::librustc_stamp;
use super::compile::libstd_stamp;
use super::compile::run_cargo;
use super::compile::rustc_cargo;
use super::compile::std_cargo;
use super::tool::prepare_tool_cargo;
use super::tool::SourceType;

// Disable the most spammy clippy lints
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
        let crates = run.make_run_crates(Alias::Library);
        run.builder.ensure(Std { target: run.target, crates });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.update_submodule(&Path::new("library").join("stdarch"));

        let target = self.target;
        let compiler = builder.compiler(builder.top_stage, builder.config.build);

        let mut cargo =
            builder::Cargo::new(builder, compiler, Mode::Std, SourceType::InTree, target, "clippy");

        std_cargo(builder, target, compiler.stage, &mut cargo);

        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let _guard = builder.msg_clippy(
            format_args!("library artifacts{}", crate_description(&self.crates)),
            target,
        );

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
            builder.ensure(check::Std::new(target));
        }

        let mut cargo = builder::Cargo::new(
            builder,
            compiler,
            Mode::Rustc,
            SourceType::InTree,
            target,
            "clippy",
        );

        rustc_cargo(builder, &mut cargo, target, compiler.stage);

        // Explicitly pass -p for all compiler crates -- this will force cargo
        // to also lint the tests/benches/examples for these crates, rather
        // than just the leaf crate.
        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let _guard = builder.msg_clippy(
            format_args!("compiler artifacts{}", crate_description(&self.crates)),
            target,
        );

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
        $name:ident, $path:expr, $tool_name:expr
        $(,is_external_tool = $external:expr)*
        $(,is_unstable_tool = $unstable:expr)*
        $(,allow_features = $allow_features:expr)?
        ;
    )+) => {
        $(

        #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            pub target: TargetSelection,
        }

        impl Step for $name {
            type Output = ();

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

                builder.ensure(check::Rustc::new(target, builder));

                let cargo = prepare_tool_cargo(
                    builder,
                    compiler,
                    Mode::ToolRustc,
                    target,
                    "clippy",
                    $path,
                    SourceType::InTree,
                    &[],
                );

                run_cargo(
                    builder,
                    cargo,
                    lint_args(builder, &[]),
                    &libstd_stamp(builder, compiler, target),
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
    CoverageDump, "src/tools/coverage-dump", "coverage-dump";
    Tidy, "src/tools/tidy", "tidy";
    Compiletest, "src/tools/compiletest", "compiletest";
    RemoteTestServer, "src/tools/remote-test-server", "remote-test-server";
    RemoteTestClient, "src/tools/remote-test-client", "remote-test-client";
);
