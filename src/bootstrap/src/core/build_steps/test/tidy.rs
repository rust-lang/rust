//! Test step for `tidy` is somewhat special becomes it combines *two* roles:
//!
//! 1. Check code style (among other things) of *other* code in the source tree.
//! 2. Running the `tidy` tool's *self-tests*.

// FIXME(#137178): currently, these two roles are combined into *one* step, presumably to make sure
// that both steps get exercised in CI. However, this makes it annoying if you want to work on tidy
// itself and *only* want to run tidy's self-tests (e.g. for faster iteration feedback).

use std::ffi::OsStr;

use clap_complete::shells;
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::DocTests;
use crate::core::build_steps::tool::Tool;
use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::core::config::flags::get_completion;

/// Runs `src/tools/tidy` and `cargo fmt --check` to detect various style problems in the
/// repository.
///
/// (To run the tidy tool's internal tests, use the alias "tidyselftest" instead.)
// FIXME(#137178): break tidy self test out into its own step. If we still want to preserve the
// current `./x test tidy` behavior, ensure the tidy self-test step *explicitly*.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tidy;

impl Step for Tidy {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    /// Runs the `tidy` tool.
    ///
    /// This tool in `src/tools` checks up on various bits and pieces of style and
    /// otherwise just implements a few lint-like checks that are specific to the
    /// compiler itself.
    ///
    /// Once tidy passes, this step also runs `fmt --check` if tests are being run
    /// for the `dev` or `nightly` channels.
    fn run(self, builder: &Builder<'_>) {
        let mut cmd = builder.tool_cmd(Tool::Tidy);
        cmd.arg(&builder.src);
        cmd.arg(&builder.initial_cargo);
        cmd.arg(&builder.out);
        // Tidy is heavily IO constrained. Still respect `-j`, but use a higher limit if `jobs`
        // hasn't been configured.
        let jobs = builder.config.jobs.unwrap_or_else(|| {
            8 * std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get) as u32
        });
        cmd.arg(jobs.to_string());
        if builder.is_verbose() {
            cmd.arg("--verbose");
        }
        if builder.config.cmd.bless() {
            cmd.arg("--bless");
        }
        if let Some(s) = builder.config.cmd.extra_checks() {
            cmd.arg(format!("--extra-checks={s}"));
        }
        let mut args = std::env::args_os();
        if args.any(|arg| arg == OsStr::new("--")) {
            cmd.arg("--");
            cmd.args(args);
        }

        if builder.config.channel == "dev" || builder.config.channel == "nightly" {
            if !builder.config.json_output {
                builder.info("fmt check");
                if builder.initial_rustfmt().is_none() {
                    let inferred_rustfmt_dir = builder.initial_sysroot.join("bin");
                    eprintln!(
                        "\
ERROR: no `rustfmt` binary found in {PATH}
INFO: `rust.channel` is currently set to \"{CHAN}\"
HELP: if you are testing a beta branch, set `rust.channel` to \"beta\" in the `config.toml` file
HELP: to skip test's attempt to check tidiness, pass `--skip src/tools/tidy` to `x.py test`",
                        PATH = inferred_rustfmt_dir.display(),
                        CHAN = builder.config.channel,
                    );
                    crate::exit!(1);
                }
                let all = false;
                crate::core::build_steps::format::format(
                    builder,
                    !builder.config.cmd.bless(),
                    all,
                    &[],
                );
            } else {
                eprintln!(
                    "WARNING: `--json-output` is not supported on rustfmt, formatting will be skipped"
                );
            }
        }

        builder.info("tidy check");
        cmd.delay_failure().run(builder);

        builder.info("x.py completions check");
        let [bash, zsh, fish, powershell] = ["x.py.sh", "x.py.zsh", "x.py.fish", "x.py.ps1"]
            .map(|filename| builder.src.join("src/etc/completions").join(filename));
        if builder.config.cmd.bless() {
            builder.ensure(crate::core::build_steps::run::GenerateCompletions);
        } else if get_completion(shells::Bash, &bash).is_some()
            || get_completion(shells::Fish, &fish).is_some()
            || get_completion(shells::PowerShell, &powershell).is_some()
            || crate::flags::get_completion(shells::Zsh, &zsh).is_some()
        {
            eprintln!(
                "x.py completions were changed; run `x.py run generate-completions` to update them"
            );
            crate::exit!(1);
        }
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = run.builder.doc_tests != DocTests::Only;
        run.path("src/tools/tidy").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Tidy);
    }
}
