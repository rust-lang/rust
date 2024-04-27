use crate::config::{Channel, ConfigInfo};
use crate::utils::{run_command, run_command_with_output_and_env, walk_dir};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

#[derive(Default)]
struct BuildArg {
    flags: Vec<String>,
    config_info: ConfigInfo,
}

impl BuildArg {
    fn new() -> Result<Option<Self>, String> {
        let mut build_arg = Self::default();
        // We skip binary name and the `build` command.
        let mut args = std::env::args().skip(2);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--features" => {
                    if let Some(arg) = args.next() {
                        build_arg.flags.push("--features".to_string());
                        build_arg.flags.push(arg.as_str().into());
                    } else {
                        return Err(
                            "Expected a value after `--features`, found nothing".to_string()
                        );
                    }
                }
                "--help" => {
                    Self::usage();
                    return Ok(None);
                }
                arg => {
                    if !build_arg.config_info.parse_argument(arg, &mut args)? {
                        return Err(format!("Unknown argument `{}`", arg));
                    }
                }
            }
        }
        Ok(Some(build_arg))
    }

    fn usage() {
        println!(
            r#"
`build` command help:

    --features [arg]       : Add a new feature [arg]"#
        );
        ConfigInfo::show_usage();
        println!("    --help                 : Show this help");
    }
}

pub fn build_sysroot(env: &HashMap<String, String>, config: &ConfigInfo) -> Result<(), String> {
    let start_dir = Path::new("build_sysroot");
    // Cleanup for previous run
    // Clean target dir except for build scripts and incremental cache
    let _ = walk_dir(
        start_dir.join("target"),
        |dir: &Path| {
            for top in &["debug", "release"] {
                let _ = fs::remove_dir_all(dir.join(top).join("build"));
                let _ = fs::remove_dir_all(dir.join(top).join("deps"));
                let _ = fs::remove_dir_all(dir.join(top).join("examples"));
                let _ = fs::remove_dir_all(dir.join(top).join("native"));

                let _ = walk_dir(
                    dir.join(top),
                    |sub_dir: &Path| {
                        if sub_dir
                            .file_name()
                            .map(|filename| filename.to_str().unwrap().starts_with("libsysroot"))
                            .unwrap_or(false)
                        {
                            let _ = fs::remove_dir_all(sub_dir);
                        }
                        Ok(())
                    },
                    |file: &Path| {
                        if file
                            .file_name()
                            .map(|filename| filename.to_str().unwrap().starts_with("libsysroot"))
                            .unwrap_or(false)
                        {
                            let _ = fs::remove_file(file);
                        }
                        Ok(())
                    },
                );
            }
            Ok(())
        },
        |_| Ok(()),
    );

    let _ = fs::remove_file(start_dir.join("Cargo.lock"));
    let _ = fs::remove_file(start_dir.join("test_target/Cargo.lock"));
    let _ = fs::remove_dir_all(start_dir.join("sysroot"));

    // Builds libs
    let mut rustflags = env.get("RUSTFLAGS").cloned().unwrap_or_default();
    if config.sysroot_panic_abort {
        rustflags.push_str(" -Cpanic=abort -Zpanic-abort-tests");
    }
    rustflags.push_str(" -Z force-unstable-if-unmarked");
    if config.no_default_features {
        rustflags.push_str(" -Csymbol-mangling-version=v0");
    }
    let mut env = env.clone();

    let mut args: Vec<&dyn AsRef<OsStr>> = vec![&"cargo", &"build", &"--target", &config.target];

    if config.no_default_features {
        rustflags.push_str(" -Csymbol-mangling-version=v0");
        args.push(&"--no-default-features");
    }

    let channel = if config.sysroot_release_channel {
        rustflags.push_str(" -Zmir-opt-level=3");
        args.push(&"--release");
        "release"
    } else {
        "debug"
    };

    env.insert("RUSTFLAGS".to_string(), rustflags);
    run_command_with_output_and_env(&args, Some(start_dir), Some(&env))?;

    // Copy files to sysroot
    let sysroot_path = start_dir.join(format!("sysroot/lib/rustlib/{}/lib/", config.target_triple));
    fs::create_dir_all(&sysroot_path).map_err(|error| {
        format!(
            "Failed to create directory `{}`: {:?}",
            sysroot_path.display(),
            error
        )
    })?;
    let copier = |dir_to_copy: &Path| {
        // FIXME: should not use shell command!
        run_command(&[&"cp", &"-r", &dir_to_copy, &sysroot_path], None).map(|_| ())
    };
    walk_dir(
        start_dir.join(&format!("target/{}/{}/deps", config.target_triple, channel)),
        copier,
        copier,
    )?;

    // Copy the source files to the sysroot (Rust for Linux needs this).
    let sysroot_src_path = start_dir.join("sysroot/lib/rustlib/src/rust");
    fs::create_dir_all(&sysroot_src_path).map_err(|error| {
        format!(
            "Failed to create directory `{}`: {:?}",
            sysroot_src_path.display(),
            error
        )
    })?;
    run_command(
        &[
            &"cp",
            &"-r",
            &start_dir.join("sysroot_src/library/"),
            &sysroot_src_path,
        ],
        None,
    )?;

    Ok(())
}

fn build_codegen(args: &mut BuildArg) -> Result<(), String> {
    let mut env = HashMap::new();

    env.insert(
        "LD_LIBRARY_PATH".to_string(),
        args.config_info.gcc_path.clone(),
    );
    env.insert(
        "LIBRARY_PATH".to_string(),
        args.config_info.gcc_path.clone(),
    );

    if args.config_info.no_default_features {
        env.insert(
            "RUSTFLAGS".to_string(),
            "-Csymbol-mangling-version=v0".to_string(),
        );
    }

    let mut command: Vec<&dyn AsRef<OsStr>> = vec![&"cargo", &"rustc"];
    if args.config_info.channel == Channel::Release {
        command.push(&"--release");
        env.insert("CHANNEL".to_string(), "release".to_string());
        env.insert("CARGO_INCREMENTAL".to_string(), "1".to_string());
    } else {
        env.insert("CHANNEL".to_string(), "debug".to_string());
    }
    if args.config_info.no_default_features {
        command.push(&"--no-default-features");
    }
    let flags = args.flags.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    for flag in &flags {
        command.push(flag);
    }
    run_command_with_output_and_env(&command, None, Some(&env))?;

    args.config_info.setup(&mut env, false)?;

    // We voluntarily ignore the error.
    let _ = fs::remove_dir_all("target/out");
    let gccjit_target = "target/out/gccjit";
    fs::create_dir_all(gccjit_target).map_err(|error| {
        format!(
            "Failed to create directory `{}`: {:?}",
            gccjit_target, error
        )
    })?;

    println!("[BUILD] sysroot");
    build_sysroot(&env, &args.config_info)?;
    Ok(())
}

pub fn run() -> Result<(), String> {
    let mut args = match BuildArg::new()? {
        Some(args) => args,
        None => return Ok(()),
    };
    args.config_info.setup_gcc_path()?;
    build_codegen(&mut args)?;
    Ok(())
}
