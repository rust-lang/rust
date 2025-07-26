use std::collections::HashMap;
use std::ffi::OsStr;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::PathBuf;

use crate::config::ConfigInfo;
use crate::utils::{get_toolchain, rustc_toolchain_version_info, rustc_version_info};

fn args(command: &str) -> Result<Option<Vec<String>>, String> {
    // We skip the binary and the "cargo"/"rustc" option.
    if let Some("--help") = std::env::args().nth(2).as_deref() {
        usage(command);
        return Ok(None);
    }
    let args = std::env::args().skip(2).collect::<Vec<_>>();
    if args.is_empty() {
        return Err(format!(
            "Expected at least one argument for `{command}` subcommand, found none"
        ));
    }
    Ok(Some(args))
}

fn usage(command: &str) {
    println!(
        r#"
`{command}` command help:

    [args]     : Arguments to be passed to the cargo command
    --help     : Show this help
"#,
    )
}

struct RustcTools {
    env: HashMap<String, String>,
    args: Vec<String>,
    toolchain: String,
    config: ConfigInfo,
}

impl RustcTools {
    fn new(command: &str) -> Result<Option<Self>, String> {
        let Some(args) = args(command)? else { return Ok(None) };

        // We first need to go to the original location to ensure that the config setup will go as
        // expected.
        let current_dir = std::env::current_dir()
            .and_then(|path| path.canonicalize())
            .map_err(|error| format!("Failed to get current directory path: {error:?}"))?;
        let current_exe = std::env::current_exe()
            .and_then(|path| path.canonicalize())
            .map_err(|error| format!("Failed to get current exe path: {error:?}"))?;
        let mut parent_dir =
            current_exe.components().map(|comp| comp.as_os_str()).collect::<Vec<_>>();
        // We run this script from "build_system/target/release/y", so we need to remove these elements.
        for to_remove in &["y", "release", "target", "build_system"] {
            if parent_dir.last().map(|part| part == to_remove).unwrap_or(false) {
                parent_dir.pop();
            } else {
                return Err(format!(
                    "Build script not executed from `build_system/target/release/y` (in path {})",
                    current_exe.display(),
                ));
            }
        }
        let parent_dir = PathBuf::from(parent_dir.join(OsStr::new("/")));
        std::env::set_current_dir(&parent_dir).map_err(|error| {
            format!("Failed to go to `{}` folder: {:?}", parent_dir.display(), error)
        })?;

        let mut env: HashMap<String, String> = std::env::vars().collect();
        let mut config = ConfigInfo::default();
        config.setup(&mut env, false)?;
        let toolchain = get_toolchain()?;

        let toolchain_version = rustc_toolchain_version_info(&toolchain)?;
        let default_version = rustc_version_info(None)?;
        if toolchain_version != default_version {
            println!(
                "rustc_codegen_gcc is built for {} but the default rustc version is {}.",
                toolchain_version.short, default_version.short,
            );
            println!("Using {}.", toolchain_version.short);
        }

        // We go back to the original folder since we now have set up everything we needed.
        std::env::set_current_dir(&current_dir).map_err(|error| {
            format!("Failed to go back to `{}` folder: {:?}", current_dir.display(), error)
        })?;
        let toolchain = format!("+{toolchain}");
        Ok(Some(Self { toolchain, args, env, config }))
    }
}

fn exec(input: &[&dyn AsRef<OsStr>], env: &HashMap<String, String>) -> Result<(), String> {
    #[cfg(unix)]
    {
        // We use `exec` to call the `execvp` syscall instead of creating a new process where the
        // command will be executed because very few signals can actually kill a current process,
        // so if segmentation fault (SIGSEGV signal) happens and we raise to the current process,
        // it will simply do nothing and we won't have the nice error message for the shell.
        let error = crate::utils::get_command_inner(input, None, Some(env)).exec();
        eprintln!("execvp syscall failed: {error:?}");
        std::process::exit(1);
    }
    #[cfg(not(unix))]
    {
        if crate::utils::run_command_with_output_and_env_no_err(input, None, Some(env)).is_err() {
            std::process::exit(1);
        }
        Ok(())
    }
}

pub fn run_cargo() -> Result<(), String> {
    let Some(mut tools) = RustcTools::new("cargo")? else { return Ok(()) };
    let rustflags = tools.env.get("RUSTFLAGS").cloned().unwrap_or_default();
    tools.env.insert("RUSTDOCFLAGS".to_string(), rustflags);
    let mut command: Vec<&dyn AsRef<OsStr>> = vec![&"cargo", &tools.toolchain];
    for arg in &tools.args {
        command.push(arg);
    }
    exec(&command, &tools.env)
}

pub fn run_rustc() -> Result<(), String> {
    let Some(tools) = RustcTools::new("rustc")? else { return Ok(()) };
    let mut command = tools.config.rustc_command_vec();
    for arg in &tools.args {
        command.push(arg);
    }
    exec(&command, &tools.env)
}
