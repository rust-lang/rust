use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::{File, remove_dir_all};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::build;
use crate::config::{Channel, ConfigInfo};
use crate::utils::{
    create_dir, get_sysroot_dir, get_toolchain, git_clone, git_clone_root_dir, remove_file,
    run_command, run_command_with_env, run_command_with_output_and_env, rustc_version_info,
    split_args, walk_dir,
};

type Env = HashMap<String, String>;
type Runner = fn(&Env, &TestArg) -> Result<(), String>;
type Runners = HashMap<&'static str, (&'static str, Runner)>;

fn get_runners() -> Runners {
    let mut runners = HashMap::new();

    runners.insert("--test-rustc", ("Run all rustc tests", test_rustc as Runner));
    runners
        .insert("--test-successful-rustc", ("Run successful rustc tests", test_successful_rustc));
    runners.insert(
        "--test-failing-ui-pattern-tests",
        ("Run failing ui pattern tests", test_failing_ui_pattern_tests),
    );
    runners.insert("--test-failing-rustc", ("Run failing rustc tests", test_failing_rustc));
    runners.insert("--projects", ("Run the tests of popular crates", test_projects));
    runners.insert("--test-libcore", ("Run libcore tests", test_libcore));
    runners.insert("--clean", ("Empty cargo target directory", clean));
    runners.insert("--build-sysroot", ("Build sysroot", build_sysroot));
    runners.insert("--std-tests", ("Run std tests", std_tests));
    runners.insert("--asm-tests", ("Run asm tests", asm_tests));
    runners.insert("--extended-tests", ("Run extended sysroot tests", extended_sysroot_tests));
    runners.insert("--extended-rand-tests", ("Run extended rand tests", extended_rand_tests));
    runners.insert(
        "--extended-regex-example-tests",
        ("Run extended regex example tests", extended_regex_example_tests),
    );
    runners.insert("--extended-regex-tests", ("Run extended regex tests", extended_regex_tests));
    runners.insert("--mini-tests", ("Run mini tests", mini_tests));
    runners.insert("--cargo-tests", ("Run cargo tests", cargo_tests));
    runners
}

fn get_number_after_arg(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<usize, String> {
    match args.next() {
        Some(nb) if !nb.is_empty() => match usize::from_str(&nb) {
            Ok(nb) => Ok(nb),
            Err(_) => Err(format!("Expected a number after `{option}`, found `{nb}`")),
        },
        _ => Err(format!("Expected a number after `{option}`, found nothing")),
    }
}

fn show_usage() {
    println!(
        r#"
`test` command help:

    --release              : Build codegen in release mode
    --sysroot-panic-abort  : Build the sysroot without unwinding support.
    --features [arg]       : Add a new feature [arg]
    --use-system-gcc       : Use system installed libgccjit
    --build-only           : Only build rustc_codegen_gcc then exits
    --nb-parts             : Used to split rustc_tests (for CI needs)
    --current-part         : Used with `--nb-parts`, allows you to specify which parts to test"#
    );
    ConfigInfo::show_usage();
    for (option, (doc, _)) in get_runners() {
        // FIXME: Instead of using the hard-coded `23` value, better to compute it instead.
        let needed_spaces = 23_usize.saturating_sub(option.len());
        let spaces: String = std::iter::repeat_n(' ', needed_spaces).collect();
        println!("    {option}{spaces}: {doc}");
    }
    println!("    --help                 : Show this help");
}

#[derive(Default, Debug)]
struct TestArg {
    build_only: bool,
    use_system_gcc: bool,
    runners: Vec<String>,
    flags: Vec<String>,
    /// Additional arguments, to be passed to commands like `cargo test`.
    test_args: Vec<String>,
    nb_parts: Option<usize>,
    current_part: Option<usize>,
    sysroot_panic_abort: bool,
    config_info: ConfigInfo,
    sysroot_features: Vec<String>,
    keep_lto_tests: bool,
}

impl TestArg {
    fn new() -> Result<Option<Self>, String> {
        let mut test_arg = Self::default();

        // We skip binary name and the `test` command.
        let mut args = std::env::args().skip(2);
        let runners = get_runners();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--features" => match args.next() {
                    Some(feature) if !feature.is_empty() => {
                        test_arg.flags.extend_from_slice(&["--features".into(), feature]);
                    }
                    _ => {
                        return Err("Expected an argument after `--features`, found nothing".into());
                    }
                },
                "--use-system-gcc" => {
                    println!("Using system GCC");
                    test_arg.use_system_gcc = true;
                }
                "--build-only" => test_arg.build_only = true,
                "--nb-parts" => {
                    test_arg.nb_parts = Some(get_number_after_arg(&mut args, "--nb-parts")?);
                }
                "--current-part" => {
                    test_arg.current_part =
                        Some(get_number_after_arg(&mut args, "--current-part")?);
                }
                "--sysroot-panic-abort" => {
                    test_arg.sysroot_panic_abort = true;
                }
                "--keep-lto-tests" => {
                    test_arg.keep_lto_tests = true;
                }
                "--sysroot-features" => match args.next() {
                    Some(feature) if !feature.is_empty() => {
                        test_arg.sysroot_features.push(feature);
                    }
                    _ => {
                        return Err(format!("Expected an argument after `{arg}`, found nothing"));
                    }
                },
                "--help" => {
                    show_usage();
                    return Ok(None);
                }
                "--" => test_arg.test_args.extend(&mut args),
                x if runners.contains_key(x)
                    && !test_arg.runners.iter().any(|runner| runner == x) =>
                {
                    test_arg.runners.push(x.into());
                }
                arg => {
                    if !test_arg.config_info.parse_argument(arg, &mut args)? {
                        return Err(format!("Unknown option {arg}"));
                    }
                }
            }
        }
        match (test_arg.current_part, test_arg.nb_parts) {
            (Some(_), Some(_)) | (None, None) => {}
            _ => {
                return Err(
                    "If either `--current-part` or `--nb-parts` is specified, the other one \
                            needs to be specified as well!"
                        .to_string(),
                );
            }
        }
        if test_arg.config_info.no_default_features {
            test_arg.flags.push("--no-default-features".into());
        }
        Ok(Some(test_arg))
    }

    pub fn is_using_gcc_master_branch(&self) -> bool {
        !self.config_info.no_default_features
    }
}

fn build_if_no_backend(env: &Env, args: &TestArg) -> Result<(), String> {
    if args.config_info.backend.is_some() {
        return Ok(());
    }
    let mut command: Vec<&dyn AsRef<OsStr>> = vec![&"cargo", &"rustc"];
    let mut tmp_env;
    let env = if args.config_info.channel == Channel::Release {
        tmp_env = env.clone();
        tmp_env.insert("CARGO_INCREMENTAL".to_string(), "1".to_string());
        command.push(&"--release");
        &tmp_env
    } else {
        env
    };
    for flag in args.flags.iter() {
        command.push(flag);
    }
    run_command_with_output_and_env(&command, None, Some(env))
}

fn clean(_env: &Env, args: &TestArg) -> Result<(), String> {
    let _ = remove_dir_all(&args.config_info.cargo_target_dir);
    let path = Path::new(&args.config_info.cargo_target_dir).join("gccjit");
    create_dir(&path)
}

fn cargo_tests(test_env: &Env, test_args: &TestArg) -> Result<(), String> {
    // First, we call `mini_tests` to build minicore for us. This ensures we are testing with a working `minicore`,
    // and that any changes we have made affect `minicore`(since it would get rebuilt).
    mini_tests(test_env, test_args)?;
    // Then, we copy some of the env vars from `test_env`
    // We don't want to pass things like `RUSTFLAGS`, since they contain the -Zcodegen-backend flag.
    // That would force `cg_gcc` to *rebuild itself* and only then run tests, which is undesirable.
    let mut env = HashMap::new();
    env.insert(
        "LD_LIBRARY_PATH".into(),
        test_env.get("LD_LIBRARY_PATH").expect("LD_LIBRARY_PATH missing!").to_string(),
    );
    env.insert(
        "LIBRARY_PATH".into(),
        test_env.get("LIBRARY_PATH").expect("LIBRARY_PATH missing!").to_string(),
    );
    env.insert(
        "CG_RUSTFLAGS".into(),
        test_env.get("CG_RUSTFLAGS").map(|s| s.as_str()).unwrap_or("").to_string(),
    );
    // Pass all the default args + the user-specified ones.
    let mut args: Vec<&dyn AsRef<OsStr>> = vec![&"cargo", &"test"];
    args.extend(test_args.test_args.iter().map(|s| s as &dyn AsRef<OsStr>));
    run_command_with_output_and_env(&args, None, Some(&env))?;
    Ok(())
}

fn mini_tests(env: &Env, args: &TestArg) -> Result<(), String> {
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[BUILD] mini_core");
    let crate_types = if args.config_info.host_triple != args.config_info.target_triple {
        "lib"
    } else {
        "lib,dylib"
    }
    .to_string();
    let mut command = args.config_info.rustc_command_vec();
    command.extend_from_slice(&[
        &"example/mini_core.rs",
        &"--crate-name",
        &"mini_core",
        &"--crate-type",
        &crate_types,
        &"--target",
        &args.config_info.target_triple,
    ]);
    run_command_with_output_and_env(&command, None, Some(env))?;

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[BUILD] example");
    let mut command = args.config_info.rustc_command_vec();
    command.extend_from_slice(&[
        &"example/example.rs",
        &"--crate-type",
        &"lib",
        &"--target",
        &args.config_info.target_triple,
    ]);
    run_command_with_output_and_env(&command, None, Some(env))?;

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] mini_core_hello_world");
    let mut command = args.config_info.rustc_command_vec();
    command.extend_from_slice(&[
        &"example/mini_core_hello_world.rs",
        &"--crate-name",
        &"mini_core_hello_world",
        &"--crate-type",
        &"bin",
        &"-g",
        &"--target",
        &args.config_info.target_triple,
    ]);
    run_command_with_output_and_env(&command, None, Some(env))?;

    let command: &[&dyn AsRef<OsStr>] = &[
        &Path::new(&args.config_info.cargo_target_dir).join("mini_core_hello_world"),
        &"abc",
        &"bcd",
    ];
    maybe_run_command_in_vm(command, env, args)?;
    Ok(())
}

fn build_sysroot(env: &Env, args: &TestArg) -> Result<(), String> {
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[BUILD] sysroot");
    let mut config = args.config_info.clone();
    config.features.extend(args.sysroot_features.iter().cloned());
    build::build_sysroot(env, &config)?;
    Ok(())
}

// TODO(GuillaumeGomez): when rewriting in Rust, refactor with the code in tests/lang_tests_common.rs if possible.
fn maybe_run_command_in_vm(
    command: &[&dyn AsRef<OsStr>],
    env: &Env,
    args: &TestArg,
) -> Result<(), String> {
    if !args.config_info.run_in_vm {
        run_command_with_output_and_env(command, None, Some(env))?;
        return Ok(());
    }
    let vm_parent_dir = match env.get("CG_GCC_VM_DIR") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir.clone()),
        _ => std::env::current_dir().unwrap(),
    };
    let vm_dir = "vm";
    let exe_to_run = command.first().unwrap();
    let exe = Path::new(&exe_to_run);
    let exe_filename = exe.file_name().unwrap();
    let vm_home_dir = vm_parent_dir.join(vm_dir).join("home");
    let vm_exe_path = vm_home_dir.join(exe_filename);
    let inside_vm_exe_path = Path::new("/home").join(exe_filename);

    let sudo_command: &[&dyn AsRef<OsStr>] = &[&"sudo", &"cp", &exe, &vm_exe_path];
    run_command_with_env(sudo_command, None, Some(env))?;

    let mut vm_command: Vec<&dyn AsRef<OsStr>> =
        vec![&"sudo", &"chroot", &vm_dir, &"qemu-m68k-static", &inside_vm_exe_path];
    vm_command.extend_from_slice(command);
    run_command_with_output_and_env(&vm_command, Some(&vm_parent_dir), Some(env))?;
    Ok(())
}

fn std_tests(env: &Env, args: &TestArg) -> Result<(), String> {
    let cargo_target_dir = Path::new(&args.config_info.cargo_target_dir);
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] arbitrary_self_types_pointers_and_wrappers");
    let mut command = args.config_info.rustc_command_vec();
    command.extend_from_slice(&[
        &"example/arbitrary_self_types_pointers_and_wrappers.rs",
        &"--crate-name",
        &"arbitrary_self_types_pointers_and_wrappers",
        &"--crate-type",
        &"bin",
        &"--target",
        &args.config_info.target_triple,
    ]);
    run_command_with_env(&command, None, Some(env))?;
    maybe_run_command_in_vm(
        &[&cargo_target_dir.join("arbitrary_self_types_pointers_and_wrappers")],
        env,
        args,
    )?;

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] alloc_system");
    let mut command = args.config_info.rustc_command_vec();
    command.extend_from_slice(&[
        &"example/alloc_system.rs",
        &"--crate-type",
        &"lib",
        &"--target",
        &args.config_info.target_triple,
    ]);
    if args.is_using_gcc_master_branch() {
        command.extend_from_slice(&[&"--cfg", &"feature=\"master\""]);
    }
    run_command_with_env(&command, None, Some(env))?;

    // FIXME: doesn't work on m68k.
    if args.config_info.host_triple == args.config_info.target_triple {
        // FIXME: create a function "display_if_not_quiet" or something along the line.
        println!("[AOT] alloc_example");
        let mut command = args.config_info.rustc_command_vec();
        command.extend_from_slice(&[
            &"example/alloc_example.rs",
            &"--crate-type",
            &"bin",
            &"--target",
            &args.config_info.target_triple,
        ]);
        run_command_with_env(&command, None, Some(env))?;
        maybe_run_command_in_vm(&[&cargo_target_dir.join("alloc_example")], env, args)?;
    }

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] dst_field_align");
    // FIXME(antoyo): Re-add -Zmir-opt-level=2 once rust-lang/rust#67529 is fixed.
    let mut command = args.config_info.rustc_command_vec();
    command.extend_from_slice(&[
        &"example/dst-field-align.rs",
        &"--crate-name",
        &"dst_field_align",
        &"--crate-type",
        &"bin",
        &"--target",
        &args.config_info.target_triple,
    ]);
    run_command_with_env(&command, None, Some(env))?;
    maybe_run_command_in_vm(&[&cargo_target_dir.join("dst_field_align")], env, args)?;

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] std_example");
    let mut command = args.config_info.rustc_command_vec();
    command.extend_from_slice(&[
        &"example/std_example.rs",
        &"--crate-type",
        &"bin",
        &"--target",
        &args.config_info.target_triple,
    ]);
    if args.is_using_gcc_master_branch() {
        command.extend_from_slice(&[&"--cfg", &"feature=\"master\""]);
    }
    run_command_with_env(&command, None, Some(env))?;
    maybe_run_command_in_vm(
        &[&cargo_target_dir.join("std_example"), &"--target", &args.config_info.target_triple],
        env,
        args,
    )?;

    let test_flags = if let Some(test_flags) = env.get("TEST_FLAGS") {
        split_args(test_flags)?
    } else {
        Vec::new()
    };
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] subslice-patterns-const-eval");
    let mut command = args.config_info.rustc_command_vec();
    command.extend_from_slice(&[
        &"example/subslice-patterns-const-eval.rs",
        &"--crate-type",
        &"bin",
        &"--target",
        &args.config_info.target_triple,
    ]);
    for test_flag in &test_flags {
        command.push(test_flag);
    }
    run_command_with_env(&command, None, Some(env))?;
    maybe_run_command_in_vm(&[&cargo_target_dir.join("subslice-patterns-const-eval")], env, args)?;

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] track-caller-attribute");
    let mut command = args.config_info.rustc_command_vec();
    command.extend_from_slice(&[
        &"example/track-caller-attribute.rs",
        &"--crate-type",
        &"bin",
        &"--target",
        &args.config_info.target_triple,
    ]);
    for test_flag in &test_flags {
        command.push(test_flag);
    }
    run_command_with_env(&command, None, Some(env))?;
    maybe_run_command_in_vm(&[&cargo_target_dir.join("track-caller-attribute")], env, args)?;

    Ok(())
}

fn setup_rustc(env: &mut Env, args: &TestArg) -> Result<PathBuf, String> {
    let toolchain = format!(
        "+{channel}-{host}",
        channel = get_toolchain()?, // May also include date
        host = args.config_info.host_triple
    );
    let rust_dir_path = Path::new(crate::BUILD_DIR).join("rust");
    // If the repository was already cloned, command will fail, so doesn't matter.
    let _ = git_clone("https://github.com/rust-lang/rust.git", Some(&rust_dir_path), false);
    let rust_dir: Option<&Path> = Some(&rust_dir_path);
    run_command(&[&"git", &"checkout", &"--", &"tests/"], rust_dir)?;
    run_command_with_output_and_env(&[&"git", &"fetch"], rust_dir, Some(env))?;
    let rustc_commit = match rustc_version_info(env.get("RUSTC").map(|s| s.as_str()))?.commit_hash {
        Some(commit_hash) => commit_hash,
        None => return Err("Couldn't retrieve rustc commit hash".to_string()),
    };
    if rustc_commit != "unknown" {
        run_command_with_output_and_env(
            &[&"git", &"checkout", &rustc_commit],
            rust_dir,
            Some(env),
        )?;
    } else {
        run_command_with_output_and_env(&[&"git", &"checkout"], rust_dir, Some(env))?;
    }

    let cargo = String::from_utf8(
        run_command_with_env(&[&"rustup", &"which", &"cargo"], rust_dir, Some(env))?.stdout,
    )
    .map_err(|error| format!("Failed to retrieve cargo path: {error:?}"))
    .and_then(|cargo| {
        let cargo = cargo.trim().to_owned();
        if cargo.is_empty() { Err("`cargo` path is empty".to_string()) } else { Ok(cargo) }
    })?;
    let rustc = String::from_utf8(
        run_command_with_env(&[&"rustup", &toolchain, &"which", &"rustc"], rust_dir, Some(env))?
            .stdout,
    )
    .map_err(|error| format!("Failed to retrieve rustc path: {error:?}"))
    .and_then(|rustc| {
        let rustc = rustc.trim().to_owned();
        if rustc.is_empty() { Err("`rustc` path is empty".to_string()) } else { Ok(rustc) }
    })?;
    let llvm_filecheck = match run_command_with_env(
        &[
            &"bash",
            &"-c",
            &"which FileCheck-10 || \
          which FileCheck-11 || \
          which FileCheck-12 || \
          which FileCheck-13 || \
          which FileCheck-14 || \
          which FileCheck",
        ],
        rust_dir,
        Some(env),
    ) {
        Ok(cmd) => String::from_utf8_lossy(&cmd.stdout).to_string(),
        Err(_) => {
            eprintln!("Failed to retrieve LLVM FileCheck, ignoring...");
            // FIXME: the test tests/run-make/no-builtins-attribute will fail if we cannot find
            // FileCheck.
            String::new()
        }
    };
    let file_path = rust_dir_path.join("config.toml");
    std::fs::write(
        &file_path,
        format!(
            r#"change-id = 115898

[rust]
codegen-backends = []
deny-warnings = false
verbose-tests = true

[build]
cargo = "{cargo}"
local-rebuild = true
rustc = "{rustc}"

[target.x86_64-unknown-linux-gnu]
llvm-filecheck = "{llvm_filecheck}"

[llvm]
download-ci-llvm = false
"#,
            cargo = cargo,
            rustc = rustc,
            llvm_filecheck = llvm_filecheck.trim(),
        ),
    )
    .map_err(|error| format!("Failed to write into `{}`: {:?}", file_path.display(), error))?;
    Ok(rust_dir_path)
}

fn asm_tests(env: &Env, args: &TestArg) -> Result<(), String> {
    let mut env = env.clone();
    let rust_dir = setup_rustc(&mut env, args)?;
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] rustc asm test suite");

    env.insert("COMPILETEST_FORCE_STAGE0".to_string(), "1".to_string());

    let codegen_backend_path = format!(
        "{pwd}/target/{channel}/librustc_codegen_gcc.{dylib_ext}",
        pwd = std::env::current_dir()
            .map_err(|error| format!("`current_dir` failed: {error:?}"))?
            .display(),
        channel = args.config_info.channel.as_str(),
        dylib_ext = args.config_info.dylib_ext,
    );

    let extra =
        if args.is_using_gcc_master_branch() { "" } else { " -Csymbol-mangling-version=v0" };

    let rustc_args = format!(
        "-Zpanic-abort-tests -Zcodegen-backend={codegen_backend_path} --sysroot {} -Cpanic=abort{extra}",
        args.config_info.sysroot_path
    );

    run_command_with_env(
        &[
            &"./x.py",
            &"test",
            &"--run",
            &"always",
            &"--stage",
            &"0",
            &"tests/assembly/asm",
            &"--compiletest-rustc-args",
            &rustc_args,
        ],
        Some(&rust_dir),
        Some(&env),
    )?;
    Ok(())
}

fn run_cargo_command(
    command: &[&dyn AsRef<OsStr>],
    cwd: Option<&Path>,
    env: &Env,
    args: &TestArg,
) -> Result<(), String> {
    run_cargo_command_with_callback(command, cwd, env, args, |cargo_command, cwd, env| {
        run_command_with_output_and_env(cargo_command, cwd, Some(env))?;
        Ok(())
    })
}

fn run_cargo_command_with_callback<F>(
    command: &[&dyn AsRef<OsStr>],
    cwd: Option<&Path>,
    env: &Env,
    args: &TestArg,
    callback: F,
) -> Result<(), String>
where
    F: Fn(&[&dyn AsRef<OsStr>], Option<&Path>, &Env) -> Result<(), String>,
{
    let toolchain = get_toolchain()?;
    let toolchain_arg = format!("+{toolchain}");
    let rustc_version = String::from_utf8(
        run_command_with_env(&[&args.config_info.rustc_command[0], &"-V"], cwd, Some(env))?.stdout,
    )
    .map_err(|error| format!("Failed to retrieve rustc version: {error:?}"))?;
    let rustc_toolchain_version = String::from_utf8(
        run_command_with_env(
            &[&args.config_info.rustc_command[0], &toolchain_arg, &"-V"],
            cwd,
            Some(env),
        )?
        .stdout,
    )
    .map_err(|error| format!("Failed to retrieve rustc +toolchain version: {error:?}"))?;

    if rustc_version != rustc_toolchain_version {
        eprintln!(
            "rustc_codegen_gcc is built for `{rustc_toolchain_version}` but the default rustc version is `{rustc_version}`.",
        );
        eprintln!("Using `{rustc_toolchain_version}`.");
    }
    let mut env = env.clone();
    let rustflags = env.get("RUSTFLAGS").cloned().unwrap_or_default();
    env.insert("RUSTDOCFLAGS".to_string(), rustflags);
    let mut cargo_command: Vec<&dyn AsRef<OsStr>> = vec![&"cargo", &toolchain_arg];
    cargo_command.extend_from_slice(command);
    callback(&cargo_command, cwd, &env)
}

// FIXME(antoyo): linker gives multiple definitions error on Linux
// echo "[BUILD] sysroot in release mode"
// ./build_sysroot/build_sysroot.sh --release

fn test_projects(env: &Env, args: &TestArg) -> Result<(), String> {
    let projects = [
        //"https://gitlab.gnome.org/GNOME/librsvg", // FIXME: doesn't compile in the CI since the
        // version of cairo and other libraries is too old.
        "https://github.com/rust-random/getrandom",
        "https://github.com/BurntSushi/memchr",
        "https://github.com/dtolnay/itoa",
        "https://github.com/rust-lang/cfg-if",
        //"https://github.com/rust-lang-nursery/lazy-static.rs", // TODO: re-enable when the
        //failing test is fixed upstream.
        //"https://github.com/marshallpierce/rust-base64", // FIXME: one test is OOM-killed.
        // TODO: ignore the base64 test that is OOM-killed.
        //"https://github.com/time-rs/time", // FIXME: one test fails (https://github.com/time-rs/time/issues/719).
        "https://github.com/rust-lang/log",
        "https://github.com/bitflags/bitflags",
        //"https://github.com/serde-rs/serde", // FIXME: one test fails.
        //"https://github.com/rayon-rs/rayon", // TODO: very slow, only run on master?
        //"https://github.com/rust-lang/cargo", // TODO: very slow, only run on master?
    ];

    let mut env = env.clone();
    let rustflags =
        format!("{} --cap-lints allow", env.get("RUSTFLAGS").cloned().unwrap_or_default());
    env.insert("RUSTFLAGS".to_string(), rustflags);
    let run_tests = |projects_path, iter: &mut dyn Iterator<Item = &&str>| -> Result<(), String> {
        for project in iter {
            let clone_result = git_clone_root_dir(project, projects_path, true)?;
            let repo_path = Path::new(&clone_result.repo_dir);
            run_cargo_command(&[&"build", &"--release"], Some(repo_path), &env, args)?;
            run_cargo_command(&[&"test"], Some(repo_path), &env, args)?;
        }

        Ok(())
    };

    let projects_path = Path::new("projects");
    create_dir(projects_path)?;

    let nb_parts = args.nb_parts.unwrap_or(0);
    if nb_parts > 0 {
        // We increment the number of tests by one because if this is an odd number, we would skip
        // one test.
        let count = projects.len() / nb_parts + 1;
        let current_part = args.current_part.unwrap();
        let start = current_part * count;
        // We remove the projects we don't want to test.
        run_tests(projects_path, &mut projects.iter().skip(start).take(count))?;
    } else {
        run_tests(projects_path, &mut projects.iter())?;
    }

    Ok(())
}

fn test_libcore(env: &Env, args: &TestArg) -> Result<(), String> {
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] libcore");
    let path = get_sysroot_dir().join("sysroot_src/library/coretests");
    let _ = remove_dir_all(path.join("target"));
    // TODO(antoyo): run in release mode when we fix the failures.
    run_cargo_command(&[&"test"], Some(&path), env, args)?;
    Ok(())
}

fn extended_rand_tests(env: &Env, args: &TestArg) -> Result<(), String> {
    if !args.is_using_gcc_master_branch() {
        println!("Not using GCC master branch. Skipping `extended_rand_tests`.");
        return Ok(());
    }
    let mut env = env.clone();
    // newer aho_corasick versions throw a deprecation warning
    let rustflags =
        format!("{} --cap-lints warn", env.get("RUSTFLAGS").cloned().unwrap_or_default());
    env.insert("RUSTFLAGS".to_string(), rustflags);

    let path = Path::new(crate::BUILD_DIR).join("rand");
    run_cargo_command(&[&"clean"], Some(&path), &env, args)?;
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] rust-random/rand");
    run_cargo_command(&[&"test", &"--workspace"], Some(&path), &env, args)?;
    Ok(())
}

fn extended_regex_example_tests(env: &Env, args: &TestArg) -> Result<(), String> {
    if !args.is_using_gcc_master_branch() {
        println!("Not using GCC master branch. Skipping `extended_regex_example_tests`.");
        return Ok(());
    }
    let path = Path::new(crate::BUILD_DIR).join("regex");
    run_cargo_command(&[&"clean"], Some(&path), env, args)?;
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] rust-lang/regex example shootout-regex-dna");
    let mut env = env.clone();
    // newer aho_corasick versions throw a deprecation warning
    let rustflags =
        format!("{} --cap-lints warn", env.get("RUSTFLAGS").cloned().unwrap_or_default());
    env.insert("RUSTFLAGS".to_string(), rustflags);
    // Make sure `[codegen mono items] start` doesn't poison the diff
    run_cargo_command(&[&"build", &"--example", &"shootout-regex-dna"], Some(&path), &env, args)?;

    run_cargo_command_with_callback(
        &[&"run", &"--example", &"shootout-regex-dna"],
        Some(&path),
        &env,
        args,
        |cargo_command, cwd, env| {
            // FIXME: rewrite this with `child.stdin.write_all()` because
            // `examples/regexdna-input.txt` is very small.
            let mut command: Vec<&dyn AsRef<OsStr>> = vec![&"bash", &"-c"];
            let cargo_args =
                cargo_command.iter().map(|s| s.as_ref().to_str().unwrap()).collect::<Vec<_>>();
            let bash_command = format!(
                "cat examples/regexdna-input.txt | {} | grep -v 'Spawned thread' > res.txt",
                cargo_args.join(" "),
            );
            command.push(&bash_command);
            run_command_with_output_and_env(&command, cwd, Some(env))?;
            run_command_with_output_and_env(
                &[&"diff", &"-u", &"res.txt", &"examples/regexdna-output.txt"],
                cwd,
                Some(env),
            )?;
            Ok(())
        },
    )?;

    Ok(())
}

fn extended_regex_tests(env: &Env, args: &TestArg) -> Result<(), String> {
    if !args.is_using_gcc_master_branch() {
        println!("Not using GCC master branch. Skipping `extended_regex_tests`.");
        return Ok(());
    }
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] rust-lang/regex tests");
    let mut env = env.clone();
    // newer aho_corasick versions throw a deprecation warning
    let rustflags =
        format!("{} --cap-lints warn", env.get("RUSTFLAGS").cloned().unwrap_or_default());
    env.insert("RUSTFLAGS".to_string(), rustflags);
    let path = Path::new(crate::BUILD_DIR).join("regex");
    run_cargo_command(
        &[
            &"test",
            &"--tests",
            &"--",
            // FIXME: try removing `--exclude-should-panic` argument
            &"--exclude-should-panic",
            &"--test-threads",
            &"1",
            &"-Zunstable-options",
            &"-q",
        ],
        Some(&path),
        &env,
        args,
    )?;
    Ok(())
}

fn extended_sysroot_tests(env: &Env, args: &TestArg) -> Result<(), String> {
    // pushd simple-raytracer
    // echo "[BENCH COMPILE] ebobby/simple-raytracer"
    // hyperfine --runs "${RUN_RUNS:-10}" --warmup 1 --prepare "cargo clean" \
    // "RUSTC=rustc RUSTFLAGS='' cargo build" \
    // "../y.sh cargo build"

    // echo "[BENCH RUN] ebobby/simple-raytracer"
    // cp ./target/debug/main ./raytracer_cg_gcc
    // hyperfine --runs "${RUN_RUNS:-10}" ./raytracer_cg_llvm ./raytracer_cg_gcc
    // popd
    extended_rand_tests(env, args)?;
    extended_regex_example_tests(env, args)?;
    extended_regex_tests(env, args)?;

    Ok(())
}

fn valid_ui_error_pattern_test(file: &str) -> bool {
    // contains //~ERROR, but shouldn't be removed
    [
        "issues/auxiliary/issue-3136-a.rs",
        "type-alias-impl-trait/auxiliary/cross_crate_ice.rs",
        "type-alias-impl-trait/auxiliary/cross_crate_ice2.rs",
        "macros/rfc-2011-nicer-assert-messages/auxiliary/common.rs",
        "imports/ambiguous-1.rs",
        "imports/ambiguous-4-extern.rs",
        "entry-point/auxiliary/bad_main_functions.rs",
    ]
    .iter()
    .any(|to_ignore| file.ends_with(to_ignore))
}

fn contains_ui_error_patterns(file_path: &Path, keep_lto_tests: bool) -> Result<bool, String> {
    // Tests generating errors.
    let file = File::open(file_path)
        .map_err(|error| format!("Failed to read `{}`: {:?}", file_path.display(), error))?;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if [
            "//@ error-pattern:",
            "//@ build-fail",
            "//@ run-fail",
            "//@ known-bug",
            "-Cllvm-args",
            "//~",
            "thread",
        ]
        .iter()
        .any(|check| line.contains(check))
        {
            return Ok(true);
        }

        if !keep_lto_tests
            && (line.contains("-Clto")
                || line.contains("-C lto")
                || line.contains("compile-flags: -Clinker-plugin-lto"))
            && !line.contains("-Clto=thin")
        {
            return Ok(true);
        }

        if line.contains("//[") && line.contains("]~") {
            return Ok(true);
        }
    }
    let file_path = file_path.display().to_string();
    if file_path.contains("ambiguous-4-extern.rs") {
        eprintln!("nothing found for {file_path:?}");
    }
    // The files in this directory contain errors.
    if file_path.contains("/error-emitter/") {
        return Ok(true);
    }
    Ok(false)
}

// # Parameters
//
// * `env`: An environment variable that provides context for the function.
// * `args`: The arguments passed to the test. This could include things like the flags, config etc.
// * `prepare_files_callback`: A callback function that prepares the files needed for the test. Its used to remove/retain tests giving Error to run various rust test suits.
// * `run_error_pattern_test`: A boolean that determines whether to run only error pattern tests.
// * `test_type`: A string that indicates the type of the test being run.
//
fn test_rustc_inner<F>(
    env: &Env,
    args: &TestArg,
    prepare_files_callback: F,
    run_error_pattern_test: bool,
    test_type: &str,
) -> Result<(), String>
where
    F: Fn(&Path) -> Result<bool, String>,
{
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] rust-lang/rust");
    let mut env = env.clone();
    let rust_path = setup_rustc(&mut env, args)?;

    if !prepare_files_callback(&rust_path)? {
        // FIXME: create a function "display_if_not_quiet" or something along the line.
        println!("Keeping all {test_type} tests");
    }

    if test_type == "ui" {
        if run_error_pattern_test {
            // After we removed the error tests that are known to panic with rustc_codegen_gcc, we now remove the passing tests since this runs the error tests.
            walk_dir(
                rust_path.join("tests/ui"),
                &mut |_dir| Ok(()),
                &mut |file_path| {
                    if contains_ui_error_patterns(file_path, args.keep_lto_tests)? {
                        Ok(())
                    } else {
                        remove_file(file_path).map_err(|e| e.to_string())
                    }
                },
                true,
            )?;
        } else {
            walk_dir(
                rust_path.join("tests/ui"),
                &mut |dir| {
                    let dir_name = dir.file_name().and_then(|name| name.to_str()).unwrap_or("");
                    if [
                        "abi",
                        "extern",
                        "unsized-locals",
                        "proc-macro",
                        "threads-sendsync",
                        "borrowck",
                        "test-attrs",
                    ]
                    .contains(&dir_name)
                    {
                        remove_dir_all(dir).map_err(|error| {
                            format!("Failed to remove folder `{}`: {:?}", dir.display(), error)
                        })?;
                    }
                    Ok(())
                },
                &mut |_| Ok(()),
                false,
            )?;

            // These two functions are used to remove files that are known to not be working currently
            // with the GCC backend to reduce noise.
            fn dir_handling(keep_lto_tests: bool) -> impl Fn(&Path) -> Result<(), String> {
                move |dir| {
                    if dir.file_name().map(|name| name == "auxiliary").unwrap_or(true) {
                        return Ok(());
                    }

                    walk_dir(
                        dir,
                        &mut dir_handling(keep_lto_tests),
                        &mut file_handling(keep_lto_tests),
                        false,
                    )
                }
            }

            fn file_handling(keep_lto_tests: bool) -> impl Fn(&Path) -> Result<(), String> {
                move |file_path| {
                    if !file_path.extension().map(|extension| extension == "rs").unwrap_or(false) {
                        return Ok(());
                    }
                    let path_str = file_path.display().to_string().replace("\\", "/");
                    if valid_ui_error_pattern_test(&path_str) {
                        return Ok(());
                    } else if contains_ui_error_patterns(file_path, keep_lto_tests)? {
                        return remove_file(&file_path);
                    }
                    Ok(())
                }
            }

            walk_dir(
                rust_path.join("tests/ui"),
                &mut dir_handling(args.keep_lto_tests),
                &mut file_handling(args.keep_lto_tests),
                false,
            )?;
        }
        let nb_parts = args.nb_parts.unwrap_or(0);
        if nb_parts > 0 {
            let current_part = args.current_part.unwrap();
            // FIXME: create a function "display_if_not_quiet" or something along the line.
            println!("Splitting ui_test into {nb_parts} parts (and running part {current_part})");
            let out = String::from_utf8(
                run_command(
                    &[
                        &"find",
                        &"tests/ui",
                        &"-type",
                        &"f",
                        &"-name",
                        &"*.rs",
                        &"-not",
                        &"-path",
                        &"*/auxiliary/*",
                    ],
                    Some(&rust_path),
                )?
                .stdout,
            )
            .map_err(|error| format!("Failed to retrieve output of find command: {error:?}"))?;
            let mut files = out
                .split('\n')
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>();
            // To ensure it'll be always the same sub files, we sort the content.
            files.sort();
            // We increment the number of tests by one because if this is an odd number, we would skip
            // one test.
            let count = files.len() / nb_parts + 1;
            // We remove the files we don't want to test.
            let start = current_part * count;
            for path in files.iter().skip(start).take(count) {
                remove_file(&rust_path.join(path))?;
            }
        }
    }

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] rustc {test_type} test suite");
    env.insert("COMPILETEST_FORCE_STAGE0".to_string(), "1".to_string());

    let extra =
        if args.is_using_gcc_master_branch() { "" } else { " -Csymbol-mangling-version=v0" };

    let rustc_args = format!(
        "{test_flags} -Zcodegen-backend={backend} --sysroot {sysroot}{extra}",
        test_flags = env.get("TEST_FLAGS").unwrap_or(&String::new()),
        backend = args.config_info.cg_backend_path,
        sysroot = args.config_info.sysroot_path,
        extra = extra,
    );

    env.get_mut("RUSTFLAGS").unwrap().clear();

    run_command_with_output_and_env(
        &[
            &"./x.py",
            &"test",
            &"--run",
            &"always",
            &"--stage",
            &"0",
            &format!("tests/{test_type}"),
            &"--compiletest-rustc-args",
            &rustc_args,
        ],
        Some(&rust_path),
        Some(&env),
    )?;
    Ok(())
}

fn test_rustc(env: &Env, args: &TestArg) -> Result<(), String> {
    test_rustc_inner(env, args, |_| Ok(false), false, "run-make")?;
    test_rustc_inner(env, args, |_| Ok(false), false, "ui")
}

fn test_failing_rustc(env: &Env, args: &TestArg) -> Result<(), String> {
    let result1 = test_rustc_inner(
        env,
        args,
        retain_files_callback("tests/failing-run-make-tests.txt", "run-make"),
        false,
        "run-make",
    );

    let result2 = test_rustc_inner(
        env,
        args,
        retain_files_callback("tests/failing-ui-tests.txt", "ui"),
        false,
        "ui",
    );

    result1.and(result2)
}

fn test_successful_rustc(env: &Env, args: &TestArg) -> Result<(), String> {
    test_rustc_inner(
        env,
        args,
        remove_files_callback("tests/failing-ui-tests.txt", "ui"),
        false,
        "ui",
    )?;
    test_rustc_inner(
        env,
        args,
        remove_files_callback("tests/failing-run-make-tests.txt", "run-make"),
        false,
        "run-make",
    )
}

fn test_failing_ui_pattern_tests(env: &Env, args: &TestArg) -> Result<(), String> {
    test_rustc_inner(
        env,
        args,
        remove_files_callback("tests/failing-ice-tests.txt", "ui"),
        true,
        "ui",
    )
}

fn retain_files_callback<'a>(
    file_path: &'a str,
    test_type: &'a str,
) -> impl Fn(&Path) -> Result<bool, String> + 'a {
    move |rust_path| {
        let files = std::fs::read_to_string(file_path).unwrap_or_default();
        let first_file_name = files.lines().next().unwrap_or("");
        // If the first line ends with a `/`, we treat all lines in the file as a directory.
        if first_file_name.ends_with('/') {
            // Treat as directory
            // Removing all tests.
            run_command(
                &[
                    &"find",
                    &format!("tests/{test_type}"),
                    &"-mindepth",
                    &"1",
                    &"-type",
                    &"d",
                    &"-exec",
                    &"rm",
                    &"-rf",
                    &"{}",
                    &"+",
                ],
                Some(rust_path),
            )?;
        } else {
            // Treat as file
            // Removing all tests.
            run_command(
                &[
                    &"find",
                    &format!("tests/{test_type}"),
                    &"-type",
                    &"f",
                    &"-name",
                    &"*.rs",
                    &"-not",
                    &"-path",
                    &"*/auxiliary/*",
                    &"-delete",
                ],
                Some(rust_path),
            )?;
        }

        // Putting back only the failing ones.
        if let Ok(files) = std::fs::read_to_string(file_path) {
            for file in files.split('\n').map(|line| line.trim()).filter(|line| !line.is_empty()) {
                run_command(&[&"git", &"checkout", &"--", &file], Some(rust_path))?;
            }
        } else {
            println!("Failed to read `{file_path}`, not putting back failing {test_type} tests");
        }

        Ok(true)
    }
}

fn remove_files_callback<'a>(
    file_path: &'a str,
    test_type: &'a str,
) -> impl Fn(&Path) -> Result<bool, String> + 'a {
    move |rust_path| {
        let files = std::fs::read_to_string(file_path).unwrap_or_default();
        let first_file_name = files.lines().next().unwrap_or("");
        // If the first line ends with a `/`, we treat all lines in the file as a directory.
        if first_file_name.ends_with('/') {
            // Removing the failing tests.
            if let Ok(files) = std::fs::read_to_string(file_path) {
                for file in
                    files.split('\n').map(|line| line.trim()).filter(|line| !line.is_empty())
                {
                    let path = rust_path.join(file);
                    if let Err(e) = remove_dir_all(&path) {
                        println!("Failed to remove directory `{}`: {}", path.display(), e);
                    }
                }
            } else {
                println!(
                    "Failed to read `{file_path}`, not putting back failing {test_type} tests"
                );
            }
        } else {
            // Removing the failing tests.
            if let Ok(files) = std::fs::read_to_string(file_path) {
                for file in
                    files.split('\n').map(|line| line.trim()).filter(|line| !line.is_empty())
                {
                    let path = rust_path.join(file);
                    remove_file(&path)?;
                }
            } else {
                println!("Failed to read `{file_path}`, not putting back failing ui tests");
            }
        }
        Ok(true)
    }
}

fn run_all(env: &Env, args: &TestArg) -> Result<(), String> {
    clean(env, args)?;
    mini_tests(env, args)?;
    build_sysroot(env, args)?;
    std_tests(env, args)?;
    // asm_tests(env, args)?;
    test_libcore(env, args)?;
    extended_sysroot_tests(env, args)?;
    cargo_tests(env, args)?;
    test_rustc(env, args)?;

    Ok(())
}

pub fn run() -> Result<(), String> {
    let mut args = match TestArg::new()? {
        Some(args) => args,
        None => return Ok(()),
    };
    let mut env: HashMap<String, String> = std::env::vars().collect();

    if !args.use_system_gcc {
        args.config_info.setup_gcc_path()?;
        let gcc_path = args.config_info.gcc_path.clone().expect(
            "The config module should have emitted an error if the GCC path wasn't provided",
        );
        env.insert("LIBRARY_PATH".to_string(), gcc_path.clone());
        env.insert("LD_LIBRARY_PATH".to_string(), gcc_path);
    }

    build_if_no_backend(&env, &args)?;
    if args.build_only {
        println!("Since it's build only, exiting...");
        return Ok(());
    }

    args.config_info.setup(&mut env, args.use_system_gcc)?;

    if args.runners.is_empty() {
        run_all(&env, &args)?;
    } else {
        let runners = get_runners();
        for runner in args.runners.iter() {
            runners.get(runner.as_str()).unwrap().1(&env, &args)?;
        }
    }

    Ok(())
}
