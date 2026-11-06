use std::collections::HashMap;

use crate::config::ConfigInfo;
use crate::utils::{run_command_with_env, setup_rustc_folder};

fn show_usage() {
    println!(
        r#"
`info` command help:

    --gcc-path     : Display the path where gcc will be located
    --llvm-commit  : Display the LLVM submodule commit rustc is using"#
    );
    ConfigInfo::show_usage();
    println!("    --help         : Show this help");
}

#[derive(Default)]
struct Info {
    gcc_path: bool,
    llvm_commit: bool,
    config: ConfigInfo,
}

impl Info {
    fn new() -> Result<Option<Self>, String> {
        let mut info = Info::default();
        info.config.no_download = true;

        // We skip binary name and the `info` command.
        let mut args = std::env::args().skip(2);
        while let Some(arg) = args.next() {
            match &*arg {
                "--help" => {
                    show_usage();
                    return Ok(None);
                }
                "--gcc-path" => info.gcc_path = true,
                "--llvm-commit" => info.llvm_commit = true,
                arg => {
                    if !info.config.parse_argument(arg, &mut args)? {
                        return Err(format!(
                            "Unknown argument {arg:?}. Use `--help` to see the available options"
                        ));
                    }
                }
            }
        }
        Ok(Some(info))
    }
}

fn show_llvm_commit() -> Result<(), String> {
    let env: HashMap<String, String> = std::env::vars().collect();
    let rust_dir_path = setup_rustc_folder(&env)?;

    let commit = String::from_utf8(
        run_command_with_env(
            &[&"git", &"rev-parse", &"HEAD:src/llvm-project"],
            Some(&rust_dir_path),
            None,
        )?
        .stdout,
    )
    .map_err(|error| format!("Invalid `git rev-parse HEAD:src/llvm-project` output: {error:?}"))?;
    println!("LLVM commit: {commit}");
    Ok(())
}

pub fn run() -> Result<(), String> {
    let Some(mut info) = Info::new()? else { return Ok(()) };

    if info.gcc_path {
        info.config.setup_gcc_path()?;
        if let Some(gcc_path) = info.config.gcc_path {
            println!("GCC path: {gcc_path}");
        }
    }
    if info.llvm_commit {
        show_llvm_commit()?;
    }
    Ok(())
}
