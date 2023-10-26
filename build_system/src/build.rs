use crate::config::{set_config, ConfigInfo};
use crate::utils::{
    get_gcc_path, run_command, run_command_with_output_and_env, walk_dir,
};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

#[derive(Default)]
struct BuildArg {
    codegen_release_channel: bool,
    sysroot_release_channel: bool,
    sysroot_panic_abort: bool,
    flags: Vec<String>,
    gcc_path: String,
}

impl BuildArg {
    fn new() -> Result<Option<Self>, String> {
        let gcc_path = get_gcc_path()?;
        let mut build_arg = Self {
            gcc_path,
            ..Default::default()
        };
        // We skip binary name and the `build` command.
        let mut args = std::env::args().skip(2);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--release" => build_arg.codegen_release_channel = true,
                "--release-sysroot" => build_arg.sysroot_release_channel = true,
                "--no-default-features" => {
                    build_arg.flags.push("--no-default-features".to_string());
                }
                "--sysroot-panic-abort" => {
                    build_arg.sysroot_panic_abort = true;
                },
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
                "--target-triple" => {
                    if args.next().is_some() {
                        // Handled in config.rs.
                    } else {
                        return Err(
                            "Expected a value after `--target-triple`, found nothing".to_string()
                        );
                    }
                }
                "--target" => {
                    if args.next().is_some() {
                        // Handled in config.rs.
                    } else {
                        return Err(
                            "Expected a value after `--target`, found nothing".to_string()
                        );
                    }
                }
                arg => return Err(format!("Unknown argument `{}`", arg)),
            }
        }
        Ok(Some(build_arg))
    }

    fn usage() {
        println!(
            r#"
`build` command help:

    --release              : Build codegen in release mode
    --release-sysroot      : Build sysroot in release mode
    --sysroot-panic-abort  : Build the sysroot without unwinding support.
    --no-default-features  : Add `--no-default-features` flag
    --features [arg]       : Add a new feature [arg]
    --target-triple [arg]  : Set the target triple to [arg]
    --help                 : Show this help
"#
        )
    }
}

fn build_sysroot(
    env: &mut HashMap<String, String>,
    args: &BuildArg,
    config: &ConfigInfo,
) -> Result<(), String> {
    std::env::set_current_dir("build_sysroot")
        .map_err(|error| format!("Failed to go to `build_sysroot` directory: {:?}", error))?;
    // Cleanup for previous run
    // Clean target dir except for build scripts and incremental cache
    let _ = walk_dir(
        "target",
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

    let _ = fs::remove_file("Cargo.lock");
    let _ = fs::remove_file("test_target/Cargo.lock");
    let _ = fs::remove_dir_all("sysroot");

    // Builds libs
    let mut rustflags = env
        .get("RUSTFLAGS")
        .cloned()
        .unwrap_or_default();
    if args.sysroot_panic_abort {
        rustflags.push_str(" -Cpanic=abort -Zpanic-abort-tests");
    }
    env.insert(
        "RUSTFLAGS".to_string(),
        format!("{} -Zmir-opt-level=3", rustflags),
    );
    let channel = if args.sysroot_release_channel {
        run_command_with_output_and_env(
            &[
                &"cargo",
                &"build",
                &"--target",
                &config.target,
                &"--release",
            ],
            None,
            Some(&env),
        )?;
        "release"
    } else {
        run_command_with_output_and_env(
            &[
                &"cargo",
                &"build",
                &"--target",
                &config.target,
            ],
            None,
            Some(env),
        )?;
        "debug"
    };

    // Copy files to sysroot
    let sysroot_path = format!("sysroot/lib/rustlib/{}/lib/", config.target_triple);
    fs::create_dir_all(&sysroot_path)
        .map_err(|error| format!("Failed to create directory `{}`: {:?}", sysroot_path, error))?;
    let copier = |dir_to_copy: &Path| {
        run_command(&[&"cp", &"-r", &dir_to_copy, &sysroot_path], None).map(|_| ())
    };
    walk_dir(
        &format!("target/{}/{}/deps", config.target_triple, channel),
        copier,
        copier,
    )?;

    Ok(())
}

fn build_codegen(args: &BuildArg) -> Result<(), String> {
    let mut env = HashMap::new();

    env.insert("LD_LIBRARY_PATH".to_string(), args.gcc_path.clone());
    env.insert("LIBRARY_PATH".to_string(), args.gcc_path.clone());

    let mut command: Vec<&dyn AsRef<OsStr>> = vec![&"cargo", &"rustc"];
    if args.codegen_release_channel {
        command.push(&"--release");
        env.insert("CHANNEL".to_string(), "release".to_string());
        env.insert("CARGO_INCREMENTAL".to_string(), "1".to_string());
    } else {
        env.insert("CHANNEL".to_string(), "debug".to_string());
    }
    let flags = args.flags.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    for flag in &flags {
        command.push(flag);
    }
    run_command_with_output_and_env(&command, None, Some(&env))?;

    let config = set_config(&mut env, &[], Some(&args.gcc_path))?;

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
    build_sysroot(
        &mut env,
        args,
        &config,
    )?;
    Ok(())
}

pub fn run() -> Result<(), String> {
    let args = match BuildArg::new()? {
        Some(args) => args,
        None => return Ok(()),
    };
    build_codegen(&args)?;
    Ok(())
}
