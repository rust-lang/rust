use crate::config::ConfigInfo;
use crate::utils::{get_gcc_path, run_command, run_command_with_output_and_env, walk_dir};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

#[derive(Default)]
struct BuildArg {
    codegen_release_channel: bool,
    flags: Vec<String>,
    gcc_path: String,
    config_info: ConfigInfo,
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
                "--no-default-features" => {
                    build_arg.flags.push("--no-default-features".to_string());
                }
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

    --release              : Build codegen in release mode
    --no-default-features  : Add `--no-default-features` flag
    --features [arg]       : Add a new feature [arg]"#
        );
        ConfigInfo::show_usage();
        println!("    --help                 : Show this help");
    }
}

fn build_sysroot_inner(
    env: &HashMap<String, String>,
    sysroot_panic_abort: bool,
    sysroot_release_channel: bool,
    config: &ConfigInfo,
    start_dir: Option<&Path>,
) -> Result<(), String> {
    let start_dir = start_dir.unwrap_or_else(|| Path::new("."));
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
    if sysroot_panic_abort {
        rustflags.push_str(" -Cpanic=abort -Zpanic-abort-tests");
    }
    rustflags.push_str(" -Z force-unstable-if-unmarked");
    let mut env = env.clone();
    let channel = if sysroot_release_channel {
        env.insert(
            "RUSTFLAGS".to_string(),
            format!("{} -Zmir-opt-level=3", rustflags),
        );
        run_command_with_output_and_env(
            &[
                &"cargo",
                &"build",
                &"--target",
                &config.target_triple,
                &"--release",
            ],
            Some(start_dir),
            Some(&env),
        )?;
        "release"
    } else {
        env.insert("RUSTFLAGS".to_string(), rustflags);

        run_command_with_output_and_env(
            &[&"cargo", &"build", &"--target", &config.target_triple],
            Some(start_dir),
            Some(&env),
        )?;
        "debug"
    };

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
    let sysroot_src_path = "sysroot/lib/rustlib/src/rust";
    fs::create_dir_all(&sysroot_src_path).map_err(|error| {
        format!(
            "Failed to create directory `{}`: {:?}",
            sysroot_src_path, error
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

pub fn build_sysroot(
    env: &HashMap<String, String>,
    sysroot_panic_abort: bool,
    sysroot_release_channel: bool,
    config: &ConfigInfo,
) -> Result<(), String> {
    build_sysroot_inner(
        env,
        sysroot_panic_abort,
        sysroot_release_channel,
        config,
        Some(Path::new("build_sysroot")),
    )
}

fn build_codegen(args: &mut BuildArg) -> Result<(), String> {
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

    args.config_info
        .setup(&mut env, &[], Some(&args.gcc_path))?;

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
        &env,
        args.config_info.sysroot_panic_abort,
        args.config_info.sysroot_release_channel,
        &args.config_info,
    )?;
    Ok(())
}

pub fn run() -> Result<(), String> {
    let mut args = match BuildArg::new()? {
        Some(args) => args,
        None => return Ok(()),
    };
    build_codegen(&mut args)?;
    Ok(())
}
