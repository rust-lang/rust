use std::ffi::OsStr;
use std::path::Path;

use crate::utils::run_command_with_output;

fn show_usage() {
    println!(
        r#"
`fmt` command help:

    --check                : Pass `--check` argument to `cargo fmt` commands
    --help                 : Show this help"#
    );
}

pub fn run() -> Result<(), String> {
    let mut check = false;
    // We skip binary name and the `info` command.
    let args = std::env::args().skip(2);
    for arg in args {
        match arg.as_str() {
            "--help" => {
                show_usage();
                return Ok(());
            }
            "--check" => check = true,
            _ => return Err(format!("Unknown option {arg}")),
        }
    }

    let cmd: &[&dyn AsRef<OsStr>] =
        if check { &[&"cargo", &"fmt", &"--check"] } else { &[&"cargo", &"fmt"] };

    run_command_with_output(cmd, Some(Path::new(".")))?;
    run_command_with_output(cmd, Some(Path::new("build_system")))
}
