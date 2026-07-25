use std::ffi::OsStr;
use std::path::Path;

use crate::utils::{
    check_exit_status, run_command_with_output, run_command_with_output_and_get_it, walk_dir,
};

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

    let (exit_status, stderr) = run_command_with_output_and_get_it(cmd, Some(Path::new(".")))?;
    if !exit_status.success() {
        let mut iter = stderr.split('\n');
        if let Some(line) = iter.next()
            && line.contains("is not installed for the toolchain")
            && let Some(line) = iter.next()
            && line.contains("run `rustup component add")
            && let Some(cmd) = line.split('`').nth(1)
        {
            println!("`rustfmt` is not installed for this toolchain, installing it...");
            // A weird round-about way to get a `&&str` so I can get a `&dyn AsRef<OsStr>` but
            // as long as it works...
            let cmd = cmd.split(' ').collect::<Vec<_>>();
            let cmd = cmd.iter().map(|s: &&str| s as &dyn AsRef<OsStr>).collect::<Vec<_>>();
            run_command_with_output(cmd.as_slice(), Some(Path::new(".")))?;
        } else {
            // If the component is installed, then it's something else. In this case we fail like we
            // should have and let the user handles the error.
            check_exit_status(cmd, Some(Path::new(".")), exit_status, None, true)?;
        }
        // We retry the command...
        run_command_with_output(cmd, Some(Path::new(".")))?;
    }
    run_command_with_output(cmd, Some(Path::new("build_system")))?;
    run_command_with_output(cmd, Some(Path::new("build_system/asm-tester")))?;

    run_rustfmt_recursively("tests/run", check)
}

fn run_rustfmt_recursively<P>(dir: P, check: bool) -> Result<(), String>
where
    P: AsRef<Path>,
{
    walk_dir(
        dir,
        &mut |dir| run_rustfmt_recursively(dir, check),
        &mut |file_path| {
            if file_path.extension().filter(|ext| ext == &OsStr::new("rs")).is_some() {
                let rustfmt_cmd: &[&dyn AsRef<OsStr>] = if check {
                    &[&"rustfmt", &"--check", &file_path]
                } else {
                    &[&"rustfmt", &file_path]
                };

                run_command_with_output(rustfmt_cmd, Some(Path::new(".")))
            } else {
                Ok(())
            }
        },
        true,
    )
}
