use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

use crate::config::{Channel, ConfigInfo};
use crate::utils::{
    copy_file, create_dir, get_sysroot_dir, run_command, run_command_with_output_and_env, walk_dir,
};

#[derive(Default)]
struct BuildArg {
    flags: Vec<String>,
    config_info: ConfigInfo,
    build_sysroot: bool,
}

impl BuildArg {
    /// Creates a new `BuildArg` instance by parsing command-line arguments.
    fn new() -> Result<Option<Self>, String> {
        let mut build_arg = Self::default();
        // Skip binary name and the `build` command.
        let mut args = std::env::args().skip(2);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--sysroot" => {
                    build_arg.build_sysroot = true;
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

    --sysroot              : Build with sysroot"#
        );
        ConfigInfo::show_usage();
        println!("    --help                 : Show this help");
    }
}

fn cleanup_sysroot_previous_build(start_dir: &Path) {
    // Cleanup for previous run
    // Clean target dir except for build scripts and incremental cache
    let _ = walk_dir(
        start_dir.join("target"),
        &mut |dir: &Path| {
            for top in &["debug", "release"] {
                let _ = fs::remove_dir_all(dir.join(top).join("build"));
                let _ = fs::remove_dir_all(dir.join(top).join("deps"));
                let _ = fs::remove_dir_all(dir.join(top).join("examples"));
                let _ = fs::remove_dir_all(dir.join(top).join("native"));

                let _ = walk_dir(
                    dir.join(top),
                    &mut |sub_dir: &Path| {
                        if sub_dir
                            .file_name()
                            .map(|filename| filename.to_str().unwrap().starts_with("libsysroot"))
                            .unwrap_or(false)
                        {
                            let _ = fs::remove_dir_all(sub_dir);
                        }
                        Ok(())
                    },
                    &mut |file: &Path| {
                        if file
                            .file_name()
                            .map(|filename| filename.to_str().unwrap().starts_with("libsysroot"))
                            .unwrap_or(false)
                        {
                            let _ = fs::remove_file(file);
                        }
                        Ok(())
                    },
                    false,
                );
            }
            Ok(())
        },
        &mut |_| Ok(()),
        false,
    );

    let _ = fs::remove_file(start_dir.join("Cargo.lock"));
    let _ = fs::remove_file(start_dir.join("test_target/Cargo.lock"));
    let _ = fs::remove_dir_all(start_dir.join("sysroot"));
}

pub fn create_build_sysroot_content(start_dir: &Path) -> Result<(), String> {
    if !start_dir.is_dir() {
        create_dir(start_dir)?;
    }
    copy_file("build_system/build_sysroot/Cargo.toml", &start_dir.join("Cargo.toml"))?;
    copy_file("build_system/build_sysroot/Cargo.lock", &start_dir.join("Cargo.lock"))?;

    let src_dir = start_dir.join("src");
    if !src_dir.is_dir() {
        create_dir(&src_dir)?;
    }
    copy_file("build_system/build_sysroot/lib.rs", &start_dir.join("src/lib.rs"))
}

pub fn build_sysroot(env: &HashMap<String, String>, config: &ConfigInfo) -> Result<(), String> {
    let start_dir = get_sysroot_dir();

    cleanup_sysroot_previous_build(&start_dir);
    create_build_sysroot_content(&start_dir)?;

    // Builds libs
    let mut rustflags = env.get("RUSTFLAGS").cloned().unwrap_or_default();
    if config.sysroot_panic_abort {
        rustflags.push_str(" -Cpanic=abort -Zpanic-abort-tests");
    }
    rustflags.push_str(" -Z force-unstable-if-unmarked");
    if config.no_default_features {
        rustflags.push_str(" -Csymbol-mangling-version=v0");
    }
    rustflags.push_str(" --print link-args");

    let mut args: Vec<&dyn AsRef<OsStr>> = vec![&"cargo", &"build", &"--target", &config.target];
    for feature in &config.features {
        args.push(&"--features");
        args.push(feature);
    }

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

    if let Ok(cg_rustflags) = std::env::var("CG_RUSTFLAGS") {
        rustflags.push(' ');
        rustflags.push_str(&cg_rustflags);
    }

    let mut env = env.clone();
    /*rustflags.push_str(" -C link-arg=-Wl,--verbose");
    rustflags.push_str(" -C link-arg=-Wl,--warn-shared-textrel"); // Mold and Gold.
    //rustflags.push_str(" -C link-arg=-Wl,--warn-textrel"); // Mold
    rustflags.push_str(" -C link-arg=-Wl,--warn-unresolved-symbols");
    rustflags.push_str(" -C link-arg=-Wl,--fatal-warnings");
    rustflags.push_str(" -C link-arg=-Wl,--print-gc-sections");*/
    //rustflags.push_str(" -C link-arg=-Wl,--no-apply-dynamic-relocs"); // Mold
    /*rustflags.push_str(" -C link-arg=-Wl,--warn-ifunc-textrel");
    rustflags.push_str(" -C link-arg=-Wl,--warn-backrefs");
    rustflags.push_str(" -C link-arg=-Wl,-znotext");
    //rustflags.push_str(" -C link-arg=-Wl,--emit-relocs");
    //rustflags.push_str(" -C link-arg=-Wl,--trace-symbol=memchr::arch::x86_64::memchr::memchr_raw::FN");
    //rustflags.push_str(" -C link-arg=-Wl,--trace-symbol=_ZN6memchr4arch6x86_646memchr10memchr_raw2FN17haaf621f7b8ca567eE");
    //rustflags.push_str(" -C link-arg=-Wl,--print-map");
    //rustflags.push_str(" -C link-arg=-Wl,");*/
    //rustflags.push_str(" -Clinker=/usr/bin/ld.gold");
    // TODO: try with verbose, warnings and logs.
    //rustflags.push_str(" -Clinker=/usr/bin/clang");
    //rustflags.push_str(" -Clink-arg=--ld-path=/usr/bin/mold");
    //rustflags.push_str(" -Clink-arg=--ld-path=/usr/bin/ld.bfd");
    //rustflags.push_str(" -Clink-arg=--ld-path=/usr/bin/ld.gold");
    env.insert("RUSTC_LOG".to_string(), "rustc_codegen_ssa::back::link=info".to_string());
    env.insert("RUSTFLAGS".to_string(), rustflags);
    run_command_with_output_and_env(&args, Some(&start_dir), Some(&env))?;

    // Copy files to sysroot
    let sysroot_path = start_dir.join(format!("sysroot/lib/rustlib/{}/lib/", config.target_triple));
    create_dir(&sysroot_path)?;
    let mut copier = |dir_to_copy: &Path| {
        // FIXME: should not use shell command!
        run_command(&[&"cp", &"-r", &dir_to_copy, &sysroot_path], None).map(|_| ())
    };
    walk_dir(
        start_dir.join(&format!("target/{}/{}/deps", config.target_triple, channel)),
        &mut copier.clone(),
        &mut copier,
        false,
    )?;

    // Copy the source files to the sysroot (Rust for Linux needs this).
    let sysroot_src_path = start_dir.join("sysroot/lib/rustlib/src/rust");
    create_dir(&sysroot_src_path)?;
    run_command(&[&"cp", &"-r", &start_dir.join("sysroot_src/library/"), &sysroot_src_path], None)?;

    Ok(())
}

fn build_codegen(args: &mut BuildArg) -> Result<(), String> {
    let mut env = HashMap::new();

    env.insert("LD_LIBRARY_PATH".to_string(), args.config_info.gcc_path.clone());
    env.insert("LIBRARY_PATH".to_string(), args.config_info.gcc_path.clone());

    if args.config_info.no_default_features {
        env.insert("RUSTFLAGS".to_string(), "-Csymbol-mangling-version=v0".to_string());
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
    create_dir(gccjit_target)?;
    if args.build_sysroot {
        println!("[BUILD] sysroot");
        build_sysroot(&env, &args.config_info)?;
    }
    Ok(())
}

/// Executes the build process.
pub fn run() -> Result<(), String> {
    let mut args = match BuildArg::new()? {
        Some(args) => args,
        None => return Ok(()),
    };
    args.config_info.setup_gcc_path()?;
    build_codegen(&mut args)?;
    Ok(())
}
