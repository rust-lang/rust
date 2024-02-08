use std::env;
use std::path::PathBuf;
use std::process::{Command, Output};

pub fn aux_build(args: &str) -> Output {
    let (words, mut cmd) = build_common(args);
    cmd.arg("--crate-type=lib");
    for word in words {
        cmd.arg(word);
    }
    let output = cmd.output().unwrap();
    if !output.status.success() {
        handle_failed_output(&format!("{:?}", cmd), output);
    }
    output
}

fn build_common(args: &str) -> (Vec<String>, Command) {
    let rustc = env::var("RUSTC").unwrap();
    let words = shell_words::split(args).expect("failed to parse arguments");
    let mut cmd = Command::new(rustc);
    cmd.arg("--out-dir")
        .arg(env::var("TMPDIR").unwrap())
        .arg("-L")
        .arg(env::var("TMPDIR").unwrap());
    (words, cmd)
}

fn handle_failed_output(cmd: &str, output: Output) -> ! {
    println!("command failed: `{}`", cmd);
    println!("=== STDOUT ===\n{}\n\n", String::from_utf8(output.stdout).unwrap());
    println!("=== STDERR ===\n{}\n\n", String::from_utf8(output.stderr).unwrap());
    std::process::exit(1)
}

pub fn rustc(args: &str) -> Output {
    let (words, mut cmd) = build_common(args);
    for word in words {
        cmd.arg(word);
    }
    let output = cmd.output().unwrap();
    if !output.status.success() {
        handle_failed_output(&format!("{:?}", cmd), output);
    }
    output
}

fn run_common(bin_name: &str) -> (Command, Output) {
    let mut bin_path = PathBuf::new();
    bin_path.push(std::env::var("TMPDIR").unwrap());
    bin_path.push(&bin_name);
    let ld_lib_path_envvar = std::env::var("LD_LIB_PATH_ENVVAR").unwrap();
    let mut cmd = Command::new(bin_path);
    cmd.env(&ld_lib_path_envvar, {
        let mut target_rpath_env_path = String::new();
        target_rpath_env_path.push_str(&std::env::var("TMPDIR").unwrap());
        target_rpath_env_path.push(':');
        target_rpath_env_path.push_str(&std::env::var("TARGET_RPATH_DIR").unwrap());
        target_rpath_env_path.push(':');
        target_rpath_env_path.push_str(&std::env::var(&ld_lib_path_envvar).unwrap());
        target_rpath_env_path
    });
    let output = cmd.output().unwrap();
    (cmd, output)
}

/// Run a built binary and make sure it succeeds.
pub fn run(bin_name: &str) -> Output {
    let (cmd, output) = run_common(bin_name);
    if !output.status.success() {
        handle_failed_output(&format!("{:?}", cmd), output);
    }
    output
}

/// Run a built binary and make sure it fails.
pub fn run_fail(bin_name: &str) -> Output {
    let (cmd, output) = run_common(bin_name);
    if output.status.success() {
        handle_failed_output(&format!("{:?}", cmd), output);
    }
    output
}
