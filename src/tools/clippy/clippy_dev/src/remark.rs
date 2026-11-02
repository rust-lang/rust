use std::process::Command;

/// Runs a command and exits with an error if it fails to start or execute.
fn run_command(label: &str, cmd: &mut Command) {
    match cmd.status() {
        Ok(status) => {
            if !status.success() {
                std::process::exit(status.code().unwrap_or(1));
            }
        },
        Err(error) => {
            if error.kind() == std::io::ErrorKind::NotFound {
                eprintln!(
                    "error running `{label}`: {error}; make sure that Node.js is installed \
                     correctly and that `node` and `npm` are visible on `$PATH`."
                );
            } else {
                eprintln!("error running `{label}`: {error}");
            }

            std::process::exit(1);
        },
    }
}

/// Installs the Node.js dependencies and runs remark to check the repository's Markdown files.
pub fn run() {
    // On Windows, npm is invoked through the npm.cmd executable.
    let npm = if cfg!(windows) { "npm.cmd" } else { "npm" };

    // Install the dependencies defined in .github/package-lock.json.
    // This is equivalent to running `cd .github && npm ci`.
    run_command(npm, Command::new(npm).arg("ci").current_dir(".github"));

    // Run remark with the same arguments used in CI.
    // Invoke Node.js directly for cross-platform compatibility.
    run_command(
        "node",
        Command::new("node").args([
            ".github/node_modules/remark-cli/cli.js",
            "--rc-path",
            ".github/.remarkrc",
            "--ignore-pattern",
            "target",
            "-f",
            ".",
        ]),
    );
}
