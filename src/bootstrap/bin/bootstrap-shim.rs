use std::{
    env, io,
    process::{self, Command, ExitStatus},
};

use bootstrap::{Flags, MinimalConfig};

#[path = "../../../src/tools/x/src/main.rs"]
mod x;

/// We are planning to exclude python logic from x script by executing bootstrap-shim
/// immediately. Since `find_and_run_available_bootstrap_script` executes x script first,
/// any changes on bootstrap will not be seen. To prevent this problem, in bootstrap-shim
/// we want to call the python script directly.
fn find_and_run_py_bootstrap_script() {
    #[cfg(unix)]
    fn exec_or_status(command: &mut Command) -> io::Result<ExitStatus> {
        use std::os::unix::process::CommandExt;
        Err(command.exec())
    }

    #[cfg(not(unix))]
    fn exec_or_status(command: &mut Command) -> io::Result<ExitStatus> {
        command.status()
    }

    let current_path = match env::current_dir() {
        Ok(dir) => dir,
        Err(err) => {
            eprintln!("Failed to get current directory: {err}");
            process::exit(1);
        }
    };

    for dir in current_path.ancestors() {
        let candidate = dir.join("x.py");
        if candidate.exists() {
            let mut cmd: Command;
            cmd = Command::new(x::python());
            cmd.arg(&candidate).args(env::args().skip(1)).current_dir(dir);
            let result = exec_or_status(&mut cmd);

            match result {
                Err(error) => {
                    eprintln!("Failed to invoke `{:?}`: {}", cmd, error);
                }
                Ok(status) => {
                    process::exit(status.code().unwrap_or(1));
                }
            }
        }
    }
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let flags = Flags::parse(&args);

    // If there are no untracked changes to bootstrap, download it from CI.
    // Otherwise, build it from source. Use python to build to avoid duplicating the code between python and rust.
    let config = MinimalConfig::parse(&flags, None);
    let bootstrap_bin = if let Some(commit) = last_modified_bootstrap_commit(&config) {
        config.download_bootstrap(&commit)
    } else {
        return find_and_run_py_bootstrap_script();
    };

    let args: Vec<_> = std::env::args().skip(1).collect();
    println!("Running pre-compiled bootstrap binary");
    Command::new(bootstrap_bin).args(args).status().expect("failed to spawn bootstrap binairy");
}

fn last_modified_bootstrap_commit(config: &MinimalConfig) -> Option<String> {
    config.last_modified_commit(
        &["src/bootstrap", "src/tools/build_helper", "src/tools/x"],
        "download-bootstrap",
        true,
    )
}
