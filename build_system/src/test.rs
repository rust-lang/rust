use crate::build;
use crate::config::ConfigInfo;
use crate::utils::{
    get_gcc_path, get_toolchain, run_command, run_command_with_env,
    run_command_with_output_and_env, rustc_version_info, split_args, walk_dir,
};

use std::collections::{BTreeSet, HashMap};
use std::ffi::OsStr;
use std::fs::remove_dir_all;
use std::path::{Path, PathBuf};
use std::str::FromStr;

type Env = HashMap<String, String>;
type Runner = &'static dyn Fn(&Env, &TestArg) -> Result<(), String>;
type Runners = HashMap<&'static str, (&'static str, Runner)>;

fn get_runners() -> Runners {
    let mut runners = HashMap::new();

    runners.insert(
        "--test-rustc",
        ("Run all rustc tests", &test_rustc as Runner),
    );
    runners.insert(
        "--test-successful-rustc",
        ("Run successful rustc tests", &test_successful_rustc),
    );
    runners.insert(
        "--test-failing-rustc",
        ("Run failing rustc tests", &test_failing_rustc),
    );
    runners.insert("--test-libcore", ("Run libcore tests", &test_libcore));
    runners.insert("--clean-ui-tests", ("Clean ui tests", &clean_ui_tests));
    runners.insert("--clean", ("Empty cargo target directory", &clean));
    runners.insert("--build-sysroot", ("Build sysroot", &build_sysroot));
    runners.insert("--std-tests", ("Run std tests", &std_tests));
    runners.insert("--asm-tests", ("Run asm tests", &asm_tests));
    runners.insert(
        "--extended-tests",
        ("Run extended sysroot tests", &extended_sysroot_tests),
    );
    runners.insert(
        "--extended-rand-tests",
        ("Run extended rand tests", &extended_rand_tests),
    );
    runners.insert(
        "--extended-regex-example-tests",
        (
            "Run extended regex example tests",
            &extended_regex_example_tests,
        ),
    );
    runners.insert(
        "--extended-regex-tests",
        ("Run extended regex tests", &extended_regex_tests),
    );
    runners.insert("--mini-tests", ("Run mini tests", &mini_tests));

    runners
}

fn get_number_after_arg(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<usize, String> {
    match args.next() {
        Some(nb) if !nb.is_empty() => match usize::from_str(&nb) {
            Ok(nb) => Ok(nb),
            Err(_) => Err(format!(
                "Expected a number after `{}`, found `{}`",
                option, nb
            )),
        },
        _ => Err(format!(
            "Expected a number after `{}`, found nothing",
            option
        )),
    }
}

fn show_usage() {
    println!(
        r#"
`test` command help:

    --release              : Build codegen in release mode
    --release-sysroot      : Build sysroot in release mode
    --sysroot-panic-abort  : Build the sysroot without unwinding support.
    --no-default-features  : Add `--no-default-features` flag
    --features [arg]       : Add a new feature [arg]
    --use-system-gcc       : Use system installed libgccjit
    --build-only           : Only build rustc_codegen_gcc then exits
    --use-backend          : Useful only for rustc testsuite
    --nb-parts             : Used to split rustc_tests (for CI needs)
    --current-part         : Used with `--nb-parts`, allows you to specify which parts to test"#
    );
    ConfigInfo::show_usage();
    for (option, (doc, _)) in get_runners() {
        let needed_spaces = 23_usize.saturating_sub(option.len());
        let spaces: String = std::iter::repeat(' ').take(needed_spaces).collect();
        println!("    {}{}: {}", option, spaces, doc);
    }
    println!("    --help                 : Show this help");
}

#[derive(Default, PartialEq, Eq, Clone, Copy)]
enum Channel {
    #[default]
    Debug,
    Release,
}

impl Channel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

#[derive(Default)]
struct TestArg {
    no_default_features: bool,
    build_only: bool,
    gcc_path: String,
    channel: Channel,
    use_backend: bool,
    runners: BTreeSet<String>,
    flags: Vec<String>,
    backend: Option<String>,
    nb_parts: Option<usize>,
    current_part: Option<usize>,
    sysroot_panic_abort: bool,
    config_info: ConfigInfo,
}

impl TestArg {
    fn new() -> Result<Option<Self>, String> {
        let mut use_system_gcc = false;
        let mut test_arg = Self::default();

        // We skip binary name and the `test` command.
        let mut args = std::env::args().skip(2);
        let runners = get_runners();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--release" => {
                    test_arg.channel = Channel::Release;
                    test_arg.config_info.sysroot_release_channel = true;
                }
                "--release-sysroot" => test_arg.config_info.sysroot_release_channel = true,
                "--no-default-features" => {
                    // To prevent adding it more than once.
                    if !test_arg.no_default_features {
                        test_arg.flags.push("--no-default-features".into());
                    }
                    test_arg.no_default_features = true;
                }
                "--features" => match args.next() {
                    Some(feature) if !feature.is_empty() => {
                        test_arg
                            .flags
                            .extend_from_slice(&["--features".into(), feature]);
                    }
                    _ => {
                        return Err("Expected an argument after `--features`, found nothing".into())
                    }
                },
                "--use-system-gcc" => use_system_gcc = true,
                "--build-only" => test_arg.build_only = true,
                "--use-backend" => match args.next() {
                    Some(backend) if !backend.is_empty() => test_arg.backend = Some(backend),
                    _ => {
                        return Err(
                            "Expected an argument after `--use-backend`, found nothing".into()
                        )
                    }
                },
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
                "--help" => {
                    show_usage();
                    return Ok(None);
                }
                x if runners.contains_key(x) => {
                    test_arg.runners.insert(x.into());
                }
                arg => {
                    if !test_arg.config_info.parse_argument(arg, &mut args)? {
                        return Err(format!("Unknown option {}", arg));
                    }
                }
            }

            test_arg.gcc_path = if use_system_gcc {
                println!("Using system GCC");
                "gcc".to_string()
            } else {
                get_gcc_path()?
            };
        }
        Ok(Some(test_arg))
    }
}

fn build_if_no_backend(env: &Env, args: &TestArg) -> Result<(), String> {
    if args.use_backend {
        return Ok(());
    }
    let mut command: Vec<&dyn AsRef<OsStr>> = vec![&"cargo", &"rustc"];
    if args.channel == Channel::Release {
        let mut env = env.clone();
        env.insert("CARGO_INCREMENTAL".to_string(), "1".to_string());
        command.push(&"--release");
        for flag in args.flags.iter() {
            command.push(flag);
        }
        run_command_with_output_and_env(&command, None, Some(&env))
    } else {
        for flag in args.flags.iter() {
            command.push(flag);
        }
        run_command_with_output_and_env(&command, None, Some(&env))
    }
}

fn clean(_env: &Env, args: &TestArg) -> Result<(), String> {
    let _ = std::fs::remove_dir_all(&args.config_info.cargo_target_dir);
    let path = Path::new(&args.config_info.cargo_target_dir).join("gccjit");
    std::fs::create_dir_all(&path)
        .map_err(|error| format!("failed to create folder `{}`: {:?}", path.display(), error))
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
    let mut command: Vec<&dyn AsRef<OsStr>> = Vec::new();
    for arg in args.config_info.rustc_command.iter() {
        command.push(arg);
    }
    command.extend_from_slice(&[
        &"example/mini_core.rs",
        &"--crate-name",
        &"mini_core",
        &"--crate-type",
        &crate_types,
        &"--target",
        &args.config_info.target_triple,
    ]);
    run_command_with_output_and_env(&command, None, Some(&env))?;

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[BUILD] example");
    command.clear();
    for arg in args.config_info.rustc_command.iter() {
        command.push(arg);
    }
    command.extend_from_slice(&[
        &"example/example.rs",
        &"--crate-type",
        &"lib",
        &"--target",
        &args.config_info.target_triple,
    ]);
    run_command_with_output_and_env(&command, None, Some(&env))?;

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] mini_core_hello_world");
    command.clear();
    for arg in args.config_info.rustc_command.iter() {
        command.push(arg);
    }
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
    run_command_with_output_and_env(&command, None, Some(&env))?;

    let command: &[&dyn AsRef<OsStr>] = &[
        &Path::new(&args.config_info.cargo_target_dir).join("mini_core_hello_world"),
        &"abc",
        &"bcd",
    ];
    run_command_in_vm(&command, env, args)?;
    Ok(())
}

fn build_sysroot(env: &Env, args: &TestArg) -> Result<(), String> {
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[BUILD] sysroot");
    build::build_sysroot(
        env,
        args.config_info.sysroot_panic_abort,
        args.config_info.sysroot_release_channel,
        &args.config_info,
    )?;
    Ok(())
}

// TODO(GuillaumeGomez): when rewriting in Rust, refactor with the code in tests/lang_tests_common.rs if possible.
fn run_command_in_vm(
    command: &[&dyn AsRef<OsStr>],
    env: &Env,
    args: &TestArg,
) -> Result<(), String> {
    if !args.config_info.run_in_vm {
        run_command_with_env(command, None, Some(env))?;
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
        vec![&"sudo", &"chroot", &"qemu-m68k-static", &inside_vm_exe_path];
    vm_command.extend_from_slice(command);
    run_command_with_env(&vm_command, Some(&vm_parent_dir), Some(env))?;
    Ok(())
}

fn std_tests(env: &Env, args: &TestArg) -> Result<(), String> {
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] arbitrary_self_types_pointers_and_wrappers");
    let mut command: Vec<&dyn AsRef<OsStr>> = Vec::new();
    for arg in args.config_info.rustc_command.iter() {
        command.push(arg);
    }
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
    run_command_in_vm(
        &[&Path::new(&args.config_info.cargo_target_dir)
            .join("arbitrary_self_types_pointers_and_wrappers")],
        env,
        args,
    )?;

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] alloc_system");
    let mut command: Vec<&dyn AsRef<OsStr>> = Vec::new();
    for arg in args.config_info.rustc_command.iter() {
        command.push(arg);
    }
    command.extend_from_slice(&[
        &"example/alloc_system.rs",
        &"--crate-type",
        &"lib",
        &"--target",
        &args.config_info.target_triple,
    ]);
    if !args.no_default_features {
        command.extend_from_slice(&[&"--cfg", &"feature=\"master\""]);
    }
    run_command_with_env(&command, None, Some(env))?;

    // FIXME: doesn't work on m68k.
    if args.config_info.host_triple != args.config_info.target_triple {
        // FIXME: create a function "display_if_not_quiet" or something along the line.
        println!("[AOT] alloc_example");
        let mut command: Vec<&dyn AsRef<OsStr>> = Vec::new();
        for arg in args.config_info.rustc_command.iter() {
            command.push(arg);
        }
        command.extend_from_slice(&[
            &"example/alloc_example.rs",
            &"--crate-type",
            &"bin",
            &"--target",
            &args.config_info.target_triple,
        ]);
        run_command_with_env(&command, None, Some(env))?;
        run_command_in_vm(
            &[&Path::new(&args.config_info.cargo_target_dir).join("alloc_example")],
            env,
            args,
        )?;
    }

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] dst_field_align");
    // FIXME(antoyo): Re-add -Zmir-opt-level=2 once rust-lang/rust#67529 is fixed.
    let mut command: Vec<&dyn AsRef<OsStr>> = Vec::new();
    for arg in args.config_info.rustc_command.iter() {
        command.push(arg);
    }
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
    run_command_in_vm(
        &[&Path::new(&args.config_info.cargo_target_dir).join("dst_field_align")],
        env,
        args,
    )?;

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] std_example");
    let mut command: Vec<&dyn AsRef<OsStr>> = Vec::new();
    for arg in args.config_info.rustc_command.iter() {
        command.push(arg);
    }
    command.extend_from_slice(&[
        &"example/std_example.rs",
        &"--crate-type",
        &"bin",
        &"--target",
        &args.config_info.target_triple,
    ]);
    if !args.no_default_features {
        command.extend_from_slice(&[&"--cfg", &"feature=\"master\""]);
    }
    run_command_with_env(&command, None, Some(env))?;
    run_command_in_vm(
        &[
            &Path::new(&args.config_info.cargo_target_dir).join("std_example"),
            &"--target",
            &args.config_info.target_triple,
        ],
        env,
        args,
    )?;

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] subslice-patterns-const-eval");
    let mut command: Vec<&dyn AsRef<OsStr>> = Vec::new();
    for arg in args.config_info.rustc_command.iter() {
        command.push(arg);
    }
    command.extend_from_slice(&[
        &"example/subslice-patterns-const-eval.rs",
        &"--crate-type",
        &"bin",
        &"--target",
        &args.config_info.target_triple,
    ]);
    run_command_with_env(&command, None, Some(env))?;
    run_command_in_vm(
        &[&Path::new(&args.config_info.cargo_target_dir).join("subslice-patterns-const-eval")],
        env,
        args,
    )?;

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] track-caller-attribute");
    let mut command: Vec<&dyn AsRef<OsStr>> = Vec::new();
    for arg in args.config_info.rustc_command.iter() {
        command.push(arg);
    }
    command.extend_from_slice(&[
        &"example/track-caller-attribute.rs",
        &"--crate-type",
        &"bin",
        &"--target",
        &args.config_info.target_triple,
    ]);
    run_command_with_env(&command, None, Some(env))?;
    run_command_in_vm(
        &[&Path::new(&args.config_info.cargo_target_dir).join("track-caller-attribute")],
        env,
        args,
    )?;

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[AOT] mod_bench");
    let mut command: Vec<&dyn AsRef<OsStr>> = Vec::new();
    for arg in args.config_info.rustc_command.iter() {
        command.push(arg);
    }
    command.extend_from_slice(&[
        &"example/mod_bench.rs",
        &"--crate-type",
        &"bin",
        &"--target",
        &args.config_info.target_triple,
    ]);
    run_command_with_env(&command, None, Some(env))?;
    // FIXME: the compiled binary is not run. Is it normal?

    Ok(())
}

fn setup_rustc(env: &mut Env, args: &TestArg) -> Result<(), String> {
    let toolchain = get_toolchain()?;

    let rust_dir = Some(Path::new("rust"));
    // If the repository was already cloned, command will fail, so doesn't matter.
    let _ = run_command_with_output_and_env(
        &[&"git", &"clone", &"https://github.com/rust-lang/rust.git"],
        None,
        Some(env),
    );
    run_command_with_output_and_env(&[&"git", &"fetch"], rust_dir, Some(env))?;
    let rustc_commit = match rustc_version_info(env.get("RUSTC").map(|s| s.as_str()))?.commit_hash {
        Some(commit_hash) => commit_hash,
        None => return Err("Couldn't retrieve rustc commit hash".to_string()),
    };
    run_command_with_output_and_env(&[&"git", &"checkout", &rustc_commit], rust_dir, Some(env))?;
    env.insert("RUSTFLAGS".to_string(), String::new());
    let cargo = String::from_utf8(
        run_command_with_env(&[&"rustup", &"which", &"cargo"], rust_dir, Some(env))?.stdout,
    )
    .map_err(|error| format!("Failed to retrieve cargo path: {:?}", error))
    .and_then(|cargo| {
        let cargo = cargo.trim().to_owned();
        if cargo.is_empty() {
            Err(format!("`cargo` path is empty"))
        } else {
            Ok(cargo)
        }
    })?;
    let llvm_filecheck = String::from_utf8(
        run_command_with_env(
            &[
                &"bash",
                &"-c",
                &"which FileCheck-10 || \
          which FileCheck-11 || \
          which FileCheck-12 || \
          which FileCheck-13 || \
          which FileCheck-14",
            ],
            rust_dir,
            Some(env),
        )?
        .stdout,
    )
    .map_err(|error| format!("Failed to retrieve LLVM FileCheck: {:?}", error))?;
    std::fs::write(
        "rust/config.toml",
        &format!(
            r#"change-id = 115898

[rust]
codegen-backends = []
deny-warnings = false
verbose-tests = true

[build]
cargo = "{cargo}"
local-rebuild = true
rustc = "{home}/.rustup/toolchains/{toolchain}-{host_triple}/bin/rustc"

[target.x86_64-unknown-linux-gnu]
llvm-filecheck = "{llvm_filecheck}"

[llvm]
download-ci-llvm = false
"#,
            cargo = cargo.trim(),
            home = env.get("HOME").unwrap(),
            toolchain = toolchain,
            host_triple = args.config_info.host_triple,
            llvm_filecheck = llvm_filecheck.trim(),
        ),
    )
    .map_err(|error| format!("Failed to write into `rust/config.toml`: {:?}", error))?;

    let rustc_commit = match rustc_version_info(env.get("RUSTC").map(|s| s.as_str()))?.commit_hash {
        Some(commit_hash) => commit_hash,
        None => return Err("Couldn't retrieve rustc commit hash".to_string()),
    };
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("commit: {:?}", rustc_commit);
    let command: &[&dyn AsRef<OsStr>] = &[&"git", &"checkout", &rustc_commit, &"tests"];
    run_command_with_output_and_env(command, rust_dir, Some(env))?;
    Ok(())
}

fn asm_tests(env: &Env, args: &TestArg) -> Result<(), String> {
    let mut env = env.clone();
    setup_rustc(&mut env, args)?;
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] rustc asm test suite");

    env.insert("COMPILETEST_FORCE_STAGE0".to_string(), "1".to_string());

    run_command_with_env(
        &[
            &"./x.py",
            &"test",
            &"--run",
            &"always",
            &"--stage",
            &"0",
            &"tests/assembly/asm",
            &"--rustc-args",
            &format!(
                r#"-Zpanic-abort-tests -Csymbol-mangling-version=v0 \
                -Zcodegen-backend="{pwd}/target/{channel}/librustc_codegen_gcc.{dylib_ext}" \
                --sysroot "{pwd}/build_sysroot/sysroot" -Cpanic=abort"#,
                pwd = std::env::current_dir()
                    .map_err(|error| format!("`current_dir` failed: {:?}", error))?
                    .display(),
                channel = args.channel.as_str(),
                dylib_ext = args.config_info.dylib_ext,
            )
            .as_str(),
        ],
        Some(Path::new("rust")),
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
        run_command_with_output_and_env(&cargo_command, cwd, Some(env))?;
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
    let rustc_version = String::from_utf8(
        run_command_with_env(&[&args.config_info.rustc_command[0], &"-V"], cwd, Some(env))?.stdout,
    )
    .map_err(|error| format!("Failed to retrieve rustc version: {:?}", error))?;
    let rustc_toolchain_version = String::from_utf8(
        run_command_with_env(
            &[
                &args.config_info.rustc_command[0],
                &format!("+{}", toolchain),
                &"-V",
            ],
            cwd,
            Some(env),
        )?
        .stdout,
    )
    .map_err(|error| format!("Failed to retrieve rustc +toolchain version: {:?}", error))?;

    if rustc_version != rustc_toolchain_version {
        eprintln!(
            "rustc_codegen_gcc is built for `{}` but the default rustc version is `{}`.",
            rustc_toolchain_version, rustc_version,
        );
        eprintln!("Using `{}`.", rustc_toolchain_version);
    }
    let mut cargo_command: Vec<&dyn AsRef<OsStr>> = vec![&"cargo", &toolchain];
    cargo_command.extend_from_slice(&command);
    callback(&cargo_command, cwd, env)
}

// FIXME(antoyo): linker gives multiple definitions error on Linux
// echo "[BUILD] sysroot in release mode"
// ./build_sysroot/build_sysroot.sh --release

fn test_libcore(env: &Env, args: &TestArg) -> Result<(), String> {
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] libcore");
    let path = Path::new("build_sysroot/sysroot_src/library/core/tests");
    let _ = remove_dir_all(path.join("target"));
    run_cargo_command(&[&"test"], Some(path), env, args)?;
    Ok(())
}

// echo "[BENCH COMPILE] mod_bench"
//
// COMPILE_MOD_BENCH_INLINE="$RUSTC example/mod_bench.rs --crate-type bin -Zmir-opt-level=3 -O --crate-name mod_bench_inline"
// COMPILE_MOD_BENCH_LLVM_0="rustc example/mod_bench.rs --crate-type bin -Copt-level=0 -o $cargo_target_dir/mod_bench_llvm_0 -Cpanic=abort"
// COMPILE_MOD_BENCH_LLVM_1="rustc example/mod_bench.rs --crate-type bin -Copt-level=1 -o $cargo_target_dir/mod_bench_llvm_1 -Cpanic=abort"
// COMPILE_MOD_BENCH_LLVM_2="rustc example/mod_bench.rs --crate-type bin -Copt-level=2 -o $cargo_target_dir/mod_bench_llvm_2 -Cpanic=abort"
// COMPILE_MOD_BENCH_LLVM_3="rustc example/mod_bench.rs --crate-type bin -Copt-level=3 -o $cargo_target_dir/mod_bench_llvm_3 -Cpanic=abort"
//
// Use 100 runs, because a single compilations doesn't take more than ~150ms, so it isn't very slow
// hyperfine --runs ${COMPILE_RUNS:-100} "$COMPILE_MOD_BENCH_INLINE" "$COMPILE_MOD_BENCH_LLVM_0" "$COMPILE_MOD_BENCH_LLVM_1" "$COMPILE_MOD_BENCH_LLVM_2" "$COMPILE_MOD_BENCH_LLVM_3"
// echo "[BENCH RUN] mod_bench"
// hyperfine --runs ${RUN_RUNS:-10} $cargo_target_dir/mod_bench{,_inline} $cargo_target_dir/mod_bench_llvm_*

fn extended_rand_tests(env: &Env, args: &TestArg) -> Result<(), String> {
    if args.no_default_features {
        return Ok(());
    }
    let path = Path::new("rand");
    run_cargo_command(&[&"clean"], Some(path), env, args)?;
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] rust-random/rand");
    run_cargo_command(&[&"test", &"--workspace"], Some(path), env, args)?;
    Ok(())
}

fn extended_regex_example_tests(env: &Env, args: &TestArg) -> Result<(), String> {
    if args.no_default_features {
        return Ok(());
    }
    let path = Path::new("regex");
    run_cargo_command(&[&"clean"], Some(path), env, args)?;
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] rust-lang/regex example shootout-regex-dna");
    let mut env = env.clone();
    // newer aho_corasick versions throw a deprecation warning
    env.insert("CG_RUSTFLAGS".to_string(), "--cap-lints warn".to_string());
    // Make sure `[codegen mono items] start` doesn't poison the diff
    run_cargo_command(
        &[&"build", &"--example", &"shootout-regex-dna"],
        Some(path),
        &env,
        args,
    )?;

    run_cargo_command_with_callback(
        &[&"run", &"--example", &"shootout-regex-dna"],
        Some(path),
        &env,
        args,
        |cargo_command, cwd, env| {
            // FIXME: rewrite this with `child.stdin.write_all()` because
            // `examples/regexdna-input.txt` is very small.
            let mut command: Vec<&dyn AsRef<OsStr>> = vec![&"bash", &"-c"];
            let cargo_args = cargo_command
                .iter()
                .map(|s| s.as_ref().to_str().unwrap())
                .collect::<Vec<_>>();
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
    if args.no_default_features {
        return Ok(());
    }
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] rust-lang/regex tests");
    let mut env = env.clone();
    env.insert("CG_RUSTFLAGS".to_string(), "--cap-lints warn".to_string());
    run_cargo_command(
        &[
            &"test",
            &"--tests",
            &"--",
            &"--exclude-should-panic",
            &"--test-threads",
            &"1",
            &"-Zunstable-options",
            &"-q",
        ],
        Some(Path::new("regex")),
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
    // "../cargo.sh build"

    // echo "[BENCH RUN] ebobby/simple-raytracer"
    // cp ./target/debug/main ./raytracer_cg_gcc
    // hyperfine --runs "${RUN_RUNS:-10}" ./raytracer_cg_llvm ./raytracer_cg_gcc
    // popd
    extended_rand_tests(env, args)?;
    extended_regex_example_tests(env, args)?;
    extended_regex_tests(env, args)?;

    Ok(())
}

fn should_remove_ui_test(content: &str) -> bool {
    for line in content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
    {
        if [
            "// error-pattern:",
            "// build-fail",
            "// run-fail",
            "-Cllvm-args",
            "//~",
            "// ~",
        ]
        .iter()
        .any(|check| line.contains(check))
        {
            return true;
        }
    }
    false
}

fn should_not_remove_test(file: &str) -> bool {
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

fn should_remove_test(path: &Path, path_str: &str) -> bool {
    // Tests generating errors.
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.contains("thread"))
        .unwrap_or(false)
        || [
            "consts/issue-miri-1910.rs",
            // Tests generating errors.
            "consts/issue-94675.rs",
            // this test is oom-killed in the CI.
            "mir/mir_heavy/issue-miri-1910.rs",
        ]
        .iter()
        .any(|to_ignore| path_str.ends_with(to_ignore))
}

fn test_rustc_inner<F>(env: &Env, args: &TestArg, callback: F) -> Result<(), String>
where
    F: Fn() -> Result<bool, String>,
{
    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] rust-lang/rust");
    let mut env = env.clone();
    setup_rustc(&mut env, args)?;

    walk_dir(
        "rust/tests/ui",
        |dir| {
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
            .iter()
            .any(|name| *name == dir_name)
            {
                std::fs::remove_dir_all(dir).map_err(|error| {
                    format!("Failed to remove folder `{}`: {:?}", dir.display(), error)
                })?;
            }
            Ok(())
        },
        |_| Ok(()),
    )?;

    fn dir_handling(dir: &Path) -> Result<(), String> {
        walk_dir(dir, dir_handling, file_handling)
    }
    fn file_handling(file: &Path) -> Result<(), String> {
        let path_str = file.display().to_string().replace("\\", "/");
        if !path_str.ends_with(".rs") {
            return Ok(());
        } else if should_not_remove_test(&path_str) {
            return Ok(());
        } else if should_remove_test(file, &path_str) {
            return std::fs::remove_file(file)
                .map_err(|error| format!("Failed to remove `{}`: {:?}", file.display(), error));
        }
        let file_content = std::fs::read_to_string(file)
            .map_err(|error| format!("Failed to read `{}`: {:?}", file.display(), error))?;
        if should_remove_ui_test(&file_content) {
            std::fs::remove_file(file)
                .map_err(|error| format!("Failed to remove `{}`: {:?}", file.display(), error))?;
        }
        Ok(())
    }

    walk_dir("rust/tests/ui", dir_handling, file_handling)?;
    let file = "rust/tests/ui/consts/const_cmp_type_id.rs";
    std::fs::remove_file(file)
        .map_err(|error| format!("Failed to remove `{}`: {:?}", file, error))?;
    let file = "rust/tests/ui/consts/issue-73976-monomorphic.rs";
    std::fs::remove_file(file)
        .map_err(|error| format!("Failed to remove `{}`: {:?}", file, error))?;

    if !callback()? {
        // FIXME: create a function "display_if_not_quiet" or something along the line.
        println!("Keeping all UI tests");
    }

    let nb_parts = args.nb_parts.unwrap_or(0);
    if nb_parts > 0 {
        let current_part = args.current_part.unwrap_or(0);
        // FIXME: create a function "display_if_not_quiet" or something along the line.
        println!(
            "Splitting ui_test into {} parts (and running part {})",
            nb_parts, current_part
        );
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
                Some(Path::new("rust")),
            )?
            .stdout,
        )
        .map_err(|error| format!("Failed to retrieve output of find command: {:?}", error))?;
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
        let start = nb_parts * count;
        let end = start + count;
        for (pos, path) in files.iter().enumerate() {
            if pos >= start && pos <= end {
                continue;
            }
            std::fs::remove_file(path)
                .map_err(|error| format!("Failed to remove `{}`: {:?}", path, error))?;
        }
    }

    // FIXME: create a function "display_if_not_quiet" or something along the line.
    println!("[TEST] rustc test suite");
    env.insert("COMPILETEST_FORCE_STAGE0".to_string(), "1".to_string());
    let rustc_args = env
        .get("RUSTFLAGS")
        .expect("RUSTFLAGS should not be empty at this stage");
    run_command_with_output_and_env(
        &[
            &"./x.py",
            &"test",
            &"--run",
            &"always",
            &"--stage",
            &"0",
            &"tests/ui",
            &"--rustc-args",
            &rustc_args,
        ],
        Some(Path::new("rust")),
        Some(&env),
    )?;
    Ok(())
}

fn test_rustc(env: &Env, args: &TestArg) -> Result<(), String> {
    test_rustc_inner(env, args, || Ok(false))
}

fn test_failing_rustc(env: &Env, args: &TestArg) -> Result<(), String> {
    test_rustc_inner(env, args, || {
        // Removing all tests.
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
                &"-delete",
            ],
            Some(Path::new("rust")),
        )?;
        // Putting back only the failing ones.
        let path = "failing-ui-tests.txt";
        if let Ok(files) = std::fs::read_to_string(path) {
            for file in files
                .split('\n')
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
            {
                run_command(
                    &[&"git", &"checkout", &"--", &file],
                    Some(Path::new("rust")),
                )?;
            }
        } else {
            println!(
                "Failed to read `{}`, not putting back failing ui tests",
                path
            );
        }
        Ok(true)
    })
}

fn test_successful_rustc(env: &Env, args: &TestArg) -> Result<(), String> {
    test_rustc_inner(env, args, || {
        // Removing the failing tests.
        let path = "failing-ui-tests.txt";
        if let Ok(files) = std::fs::read_to_string(path) {
            for file in files
                .split('\n')
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
            {
                let path = Path::new("rust").join(file);
                std::fs::remove_file(&path).map_err(|error| {
                    format!("failed to remove `{}`: {:?}", path.display(), error)
                })?;
            }
        } else {
            println!(
                "Failed to read `{}`, not putting back failing ui tests",
                path
            );
        }
        Ok(true)
    })
}

fn clean_ui_tests(_env: &Env, _args: &TestArg) -> Result<(), String> {
    run_command(
        &[
            &"find",
            &"rust/build/x86_64-unknown-linux-gnu/test/ui/",
            &"-name",
            &"stamp",
            &"-delete",
        ],
        None,
    )?;
    Ok(())
}

fn run_all(env: &Env, args: &TestArg) -> Result<(), String> {
    clean(env, args)?;
    mini_tests(env, args)?;
    build_sysroot(env, args)?;
    std_tests(env, args)?;
    // asm_tests(env, args)?;
    test_libcore(env, args)?;
    extended_sysroot_tests(env, args)?;
    test_rustc(env, args)?;
    Ok(())
}

pub fn run() -> Result<(), String> {
    let mut args = match TestArg::new()? {
        Some(args) => args,
        None => return Ok(()),
    };
    let mut env: HashMap<String, String> = std::env::vars().collect();

    env.insert("LD_LIBRARY_PATH".to_string(), args.gcc_path.clone());
    env.insert("LIBRARY_PATH".to_string(), args.gcc_path.clone());

    build_if_no_backend(&env, &args)?;
    if args.build_only {
        println!("Since it's build only, exiting...");
        return Ok(());
    }

    let test_flags = split_args(env.get("TEST_FLAGS").unwrap_or(&String::new()));
    args.config_info
        .setup(&mut env, &test_flags, Some(&args.gcc_path))?;

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
