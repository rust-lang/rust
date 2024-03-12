use crate::config::ConfigInfo;
use crate::utils::{
    get_toolchain, run_command_with_output_and_env_no_err, rustc_toolchain_version_info,
    rustc_version_info,
};

use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::PathBuf;

fn args() -> Result<Option<Vec<String>>, String> {
    // We skip the binary and the "cargo" option.
    if let Some("--help") = std::env::args().skip(2).next().as_deref() {
        usage();
        return Ok(None);
    }
    let args = std::env::args().skip(2).collect::<Vec<_>>();
    if args.is_empty() {
        return Err(
            "Expected at least one argument for `cargo` subcommand, found none".to_string(),
        );
    }
    Ok(Some(args))
}

fn usage() {
    println!(
        r#"
`cargo` command help:

    [args]     : Arguments to be passed to the cargo command
    --help     : Show this help
"#
    )
}

pub fn run() -> Result<(), String> {
    let args = match args()? {
        Some(a) => a,
        None => return Ok(()),
    };

    // We first need to go to the original location to ensure that the config setup will go as
    // expected.
    let current_dir = std::env::current_dir()
        .and_then(|path| path.canonicalize())
        .map_err(|error| format!("Failed to get current directory path: {:?}", error))?;
    let current_exe = std::env::current_exe()
        .and_then(|path| path.canonicalize())
        .map_err(|error| format!("Failed to get current exe path: {:?}", error))?;
    let mut parent_dir = current_exe
        .components()
        .map(|comp| comp.as_os_str())
        .collect::<Vec<_>>();
    // We run this script from "build_system/target/release/y", so we need to remove these elements.
    for to_remove in &["y", "release", "target", "build_system"] {
        if parent_dir
            .last()
            .map(|part| part == to_remove)
            .unwrap_or(false)
        {
            parent_dir.pop();
        } else {
            return Err(format!(
                "Build script not executed from `build_system/target/release/y` (in path {})",
                current_exe.display(),
            ));
        }
    }
    let parent_dir = PathBuf::from(parent_dir.join(&OsStr::new("/")));
    std::env::set_current_dir(&parent_dir).map_err(|error| {
        format!(
            "Failed to go to `{}` folder: {:?}",
            parent_dir.display(),
            error
        )
    })?;

    let mut env: HashMap<String, String> = std::env::vars().collect();
    ConfigInfo::default().setup(&mut env, false)?;
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
        format!(
            "Failed to go back to `{}` folder: {:?}",
            current_dir.display(),
            error
        )
    })?;

    let rustflags = env.get("RUSTFLAGS").cloned().unwrap_or_default();
    env.insert("RUSTDOCFLAGS".to_string(), rustflags);
    let toolchain = format!("+{}", toolchain);
    let mut command: Vec<&dyn AsRef<OsStr>> = vec![&"cargo", &toolchain];
    for arg in &args {
        command.push(arg);
    }
    if run_command_with_output_and_env_no_err(&command, None, Some(&env)).is_err() {
        std::process::exit(1);
    }

    Ok(())
}
