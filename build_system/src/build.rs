use crate::config::set_config;
use crate::utils::{
    get_gcc_path, run_command, run_command_with_env, run_command_with_output_and_env, walk_dir,
};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

#[derive(Default)]
struct BuildArg {
    codegen_release_channel: bool,
    sysroot_release_channel: bool,
    features: Vec<String>,
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
                    build_arg.features.push("--no-default-features".to_string());
                }
                "--features" => {
                    if let Some(arg) = args.next() {
                        build_arg.features.push("--features".to_string());
                        build_arg.features.push(arg.as_str().into());
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
    --no-default-features  : Add `--no-default-features` flag
    --features [arg]       : Add a new feature [arg]
    --help                 : Show this help
"#
        )
    }
}

fn build_sysroot(
    env: &mut HashMap<String, String>,
    release_mode: bool,
    target_triple: &str,
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
    let channel = if release_mode {
        let rustflags = env
            .get("RUSTFLAGS")
            .cloned()
            .unwrap_or_default();
        env.insert(
            "RUSTFLAGS".to_string(),
            format!("{} -Zmir-opt-level=3", rustflags),
        );
        run_command_with_output_and_env(
            &[
                &"cargo",
                &"build",
                &"--target",
                &target_triple,
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
                &target_triple,
                &"--features",
                &"compiler_builtins/c",
            ],
            None,
            Some(env),
        )?;
        "debug"
    };

    // Copy files to sysroot
    let sysroot_path = format!("sysroot/lib/rustlib/{}/lib/", target_triple);
    fs::create_dir_all(&sysroot_path)
        .map_err(|error| format!("Failed to create directory `{}`: {:?}", sysroot_path, error))?;
    let copier = |dir_to_copy: &Path| {
        run_command(&[&"cp", &"-r", &dir_to_copy, &sysroot_path], None).map(|_| ())
    };
    walk_dir(
        &format!("target/{}/{}/deps", target_triple, channel),
        copier,
        copier,
    )?;

    Ok(())
}

fn build_codegen(args: &BuildArg) -> Result<(), String> {
    let mut env = HashMap::new();

    let current_dir =
        std::env::current_dir().map_err(|error| format!("`current_dir` failed: {:?}", error))?;
    if let Ok(rt_root) = std::env::var("RUST_COMPILER_RT_ROOT") {
        env.insert("RUST_COMPILER_RT_ROOT".to_string(), rt_root);
    } else {
        env.insert(
            "RUST_COMPILER_RT_ROOT".to_string(),
            format!("{}", current_dir.join("llvm/compiler-rt").display()),
        );
    }
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
    let ref_features = args.features.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    for feature in &ref_features {
        command.push(feature);
    }
    run_command_with_env(&command, None, Some(&env))?;

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
        args.sysroot_release_channel,
        &config.target_triple,
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
