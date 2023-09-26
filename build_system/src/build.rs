use crate::config::set_config;
use crate::utils::{get_gcc_path, run_command_with_env, run_command_with_output, walk_dir};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

#[derive(Default)]
struct BuildArg {
    codegen_release_channel: bool,
    sysroot_release_channel: bool,
    no_default_features: bool,
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
        let mut args = std::env::args().skip(2);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--release" => build_arg.codegen_release_channel = true,
                "--release-sysroot" => build_arg.sysroot_release_channel = true,
                "--no-default-features" => build_arg.no_default_features = true,
                "--features" => {
                    if let Some(arg) = args.next() {
                        build_arg.features.push(arg.as_str().into());
                    } else {
                        return Err(format!(
                            "Expected a value after `--features`, found nothing"
                        ));
                    }
                }
                "--help" => {
                    Self::usage();
                    return Ok(None);
                }
                a => return Err(format!("Unknown argument `{a}`")),
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
        .map_err(|e| format!("Failed to go to `build_sysroot` directory: {e:?}"))?;
    // Cleanup for previous run
    //     v Clean target dir except for build scripts and incremental cache
   let _e = walk_dir(
        "target",
        |dir: &Path| {
            for top in &["debug", "release"] {
                let _e = fs::remove_dir_all(dir.join(top).join("build"));
                let _e = fs::remove_dir_all(dir.join(top).join("deps"));
                let _e = fs::remove_dir_all(dir.join(top).join("examples"));
                let _e = fs::remove_dir_all(dir.join(top).join("native"));

                let _e = walk_dir(
                    dir.join(top),
                    |sub_dir: &Path| {
                        if sub_dir
                            .file_name()
                            .map(|s| s.to_str().unwrap().starts_with("libsysroot"))
                            .unwrap_or(false)
                        {
                            let _e = fs::remove_dir_all(sub_dir);
                        }
                        Ok(())
                    },
                    |file: &Path| {
                        if file
                            .file_name()
                            .map(|s| s.to_str().unwrap().starts_with("libsysroot"))
                            .unwrap_or(false)
                        {
                            let _e = fs::remove_file(file);
                        }
                        Ok(())
                    },
                );
            }
            Ok(())
        },
        |_| Ok(()),
    );

    let _e = fs::remove_file("Cargo.lock");
    let _e = fs::remove_file("test_target/Cargo.lock");
    let _e = fs::remove_dir_all("sysroot");

    // Builds libs
    let channel = if release_mode {
        let rustflags = env
            .get(&"RUSTFLAGS".to_owned())
            .cloned()
            .unwrap_or_default();
        env.insert(
            "RUSTFLAGS".to_owned(),
            format!("{rustflags} -Zmir-opt-level=3"),
        );
        run_command_with_output(
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
        run_command_with_output(
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
    let sysroot_path = format!("sysroot/lib/rustlib/{target_triple}/lib/");
    fs::create_dir_all(&sysroot_path)
        .map_err(|e| format!("Failed to create directory `{sysroot_path}`: {e:?}"))?;
    let copier = |d: &Path| run_command_with_output(&[&"cp", &"-r", &d, &sysroot_path], None, None);
    walk_dir(
        &format!("target/{target_triple}/{channel}/deps"),
        copier,
        copier,
    )?;

    Ok(())
}

fn build_codegen(args: &BuildArg) -> Result<(), String> {
    let mut env = HashMap::new();

    let current_dir =
        std::env::current_dir().map_err(|e| format!("`current_dir` failed: {e:?}"))?;
    env.insert(
        "RUST_COMPILER_RT_ROOT".to_owned(),
        format!("{}", current_dir.join("llvm/compiler-rt").display()),
    );
    env.insert("LD_LIBRARY_PATH".to_owned(), args.gcc_path.clone());
    env.insert("LIBRARY_PATH".to_owned(), args.gcc_path.clone());

    let mut command: Vec<&dyn AsRef<OsStr>> = vec![&"cargo", &"rustc"];
    if args.codegen_release_channel {
        command.push(&"--release");
        env.insert("CHANNEL".to_owned(), "release".to_owned());
        env.insert("CARGO_INCREMENTAL".to_owned(), "1".to_owned());
    } else {
        env.insert("CHANNEL".to_owned(), "debug".to_owned());
    }
    let ref_features = args.features.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    for feature in &ref_features {
        command.push(feature);
    }
    run_command_with_env(&command, None, Some(&env))?;

    let config = set_config(&mut env, &[], Some(&args.gcc_path))?;

    // We voluntarily ignore the error.
    let _e = fs::remove_dir_all("target/out");
    let gccjit_target = "target/out/gccjit";
    fs::create_dir_all(gccjit_target)
        .map_err(|e| format!("Failed to create directory `{gccjit_target}`: {e:?}"))?;

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
        Some(a) => a,
        None => return Ok(()),
    };
    build_codegen(&args)?;
    Ok(())
}
