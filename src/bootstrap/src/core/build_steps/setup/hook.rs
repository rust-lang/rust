use std::path::PathBuf;
use std::{fs, io};


use crate::core::build_steps::setup::{PromptResult, prompt_user};
use crate::core::builder::{Builder, CommandLineStep, RunConfig, ShouldRun};
use crate::core::config::Config;
use crate::utils::helpers::{self, t};

/// Installs `src/etc/pre-push.sh` as a Git hook
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Hook;

impl CommandLineStep for Hook {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("hook")
    }

    fn is_default_step(_builder: &Builder<'_>) -> bool {
        true
    }

    fn make_run(run: RunConfig<'_>) {
        if let [cmd] = &run.paths[..]
            && cmd.assert_single_path().path.as_path().as_os_str() == "hook"
        {
            run.builder.ensure(Hook);
        }
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let config = &builder.config;

        if config.dry_run() || !config.rust_info.is_managed_git_subrepository() {
            return;
        }

        t!(install_git_hook_maybe(builder, config));
    }
}

// install a git hook to automatically run tidy, if they want
fn install_git_hook_maybe(builder: &Builder<'_>, config: &Config) -> io::Result<()> {
    let git = helpers::git(Some(&config.src))
        .args(["rev-parse", "--git-common-dir"])
        .run_capture(builder)
        .stdout();
    let git = PathBuf::from(git.trim());
    let hooks_dir = git.join("hooks");
    let dst = hooks_dir.join("pre-push");
    if dst.exists() {
        // The git hook has already been set up, or the user already has a custom hook.
        return Ok(());
    }

    println!(
        "\nRust's CI will automatically fail if it doesn't pass `tidy`, the internal tool for ensuring code quality.
If you'd like, x.py can install a git hook for you that will automatically run `test tidy` before
pushing your code to ensure your code is up to par. If you decide later that this behavior is
undesirable, simply delete the `pre-push` file from .git/hooks."
    );

    if prompt_user("Would you like to install the git hook?: [y/N]")? != Some(PromptResult::Yes) {
        println!("Ok, skipping installation!");
        return Ok(());
    }
    if !hooks_dir.exists() {
        // We need to (try to) create the hooks directory first.
        let _ = fs::create_dir(hooks_dir);
    }
    let src = config.src.join("src").join("etc").join("pre-push.sh");
    match fs::hard_link(src, &dst) {
        Err(e) => {
            eprintln!(
                "ERROR: could not create hook {}: do you already have the git hook installed?\n{}",
                dst.display(),
                e
            );
            return Err(e);
        }
        Ok(_) => println!("Linked `src/etc/pre-push.sh` to `.git/hooks/pre-push`"),
    };
    Ok(())
}
