use std::path::Path;

use crate::utils::{run_command_with_output, run_tool_and_install_it_if_not_present};

fn show_usage() {
    println!(
        r#"
`clippy` command help:

    --help  : Show this help"#
    );
}

pub fn run() -> Result<(), String> {
    // We skip binary name and the `info` command.
    let args = std::env::args().skip(2);
    #[allow(clippy::never_loop)]
    for arg in args {
        match arg.as_str() {
            "--help" => {
                show_usage();
                return Ok(());
            }
            _ => return Err(format!("Unknown option {arg}")),
        }
    }

    run_tool_and_install_it_if_not_present(&[
        &"cargo",
        &"clippy",
        &"--all-targets",
        &"--",
        &"-D",
        &"warnings",
    ])?;
    run_command_with_output(
        &[
            &"cargo",
            &"clippy",
            &"--all-targets",
            &"--no-default-features",
            &"--",
            &"-D",
            &"warnings",
        ],
        Some(Path::new(".")),
    )?;
    run_command_with_output(
        &[
            &"cargo",
            &"clippy",
            &"--all-targets",
            &"--manifest-path",
            &"build_system/Cargo.toml",
            &"--",
            &"-D",
            &"warnings",
        ],
        Some(Path::new(".")),
    )?;
    Ok(())
}
