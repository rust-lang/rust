use crate::common::{
    run_command, run_command_with_env, run_command_with_output, run_command_with_output_and_env,
    write_file,
};

use std::collections::HashMap;
use std::path::Path;

// Checks that:
//
// * `cargo fmt --all` succeeds without any warnings or errors
// * `cargo fmt --all -- --check` after formatting returns success
// * `cargo test --all` still passes (formatting did not break the build)
fn check_fmt_with_all_tests(env: HashMap<&str, &str>, current_dir: &str) -> Result<(), String> {
    check_fmt_base("--all", env, current_dir)
}

// Checks that:
//
// * `cargo fmt --all` succeeds without any warnings or errors
// * `cargo fmt --all -- --check` after formatting returns success
// * `cargo test --lib` still passes (formatting did not break the build)
fn check_fmt_with_lib_tests(env: HashMap<&str, &str>, current_dir: &str) -> Result<(), String> {
    check_fmt_base("--lib", env, current_dir)
}

fn check_fmt_base(
    test_args: &str,
    env: HashMap<&str, &str>,
    current_dir: &str,
) -> Result<(), String> {
    fn check_output_does_not_contain(output: &str, needle: &str) -> Result<(), String> {
        if output.contains(needle) {
            Err(format!("`cargo fmt --all -v` contains `{needle}`"))
        } else {
            Ok(())
        }
    }

    let output =
        run_command_with_output_and_env("cargo", &["test", test_args], current_dir, &env)?.output;
    if ["build failed", "test result: FAILED."]
        .iter()
        .any(|needle| output.contains(needle))
    {
        println!("`cargo test {test_args}` failed: {output}");
        return Ok(());
    }

    let rustfmt_toml = Path::new(current_dir).join("rustfmt.toml");
    if !rustfmt_toml.is_file() {
        write_file(rustfmt_toml, "")?;
    }

    let output =
        run_command_with_output_and_env("cargo", &["fmt", "--all", "-v"], current_dir, &env)?;
    println!("{}", output.output);

    if !output.exited_successfully {
        return Err("`cargo fmt --all -v` failed".to_string());
    }

    let output = &output.output;
    check_output_does_not_contain(output, "internal error")?;
    check_output_does_not_contain(output, "internal compiler error")?;
    check_output_does_not_contain(output, "warning")?;
    check_output_does_not_contain(output, "Warning")?;

    let output = run_command_with_output_and_env(
        "cargo",
        &["fmt", "--all", "--", "--check"],
        current_dir,
        &env,
    )?;

    if !output.exited_successfully {
        return Err("`cargo fmt --all -- -v --check` failed".to_string());
    }
    let output = &output.output;
    if let Err(error) = write_file(Path::new(current_dir).join("rustfmt_check_output"), output) {
        println!("{output}");
        return Err(error);
    }

    // This command allows to ensure that no source file was modified while running the tests.
    run_command_with_env("cargo", &["test", test_args], current_dir, &env)
}

fn show_head(integration: &str) -> Result<(), String> {
    let head = run_command_with_output("git", &["rev-parse", "HEAD"], integration)?.output;
    println!("Head commit of {integration}: {head}");
    Ok(())
}

fn run_test<F: FnOnce(HashMap<&str, &str>, &str) -> Result<(), String>>(
    integration: &str,
    git_repository: String,
    env: HashMap<&str, &str>,
    test_fn: F,
) -> Result<(), String> {
    run_command_with_output("git", &["clone", "--depth=1", git_repository.as_str()], ".")?;
    show_head(integration)?;
    test_fn(env, integration)
}

pub fn runner(args: &mut impl Iterator<Item = String>) -> Result<(), String> {
    let Some(integration) = args.next() else {
        return Err("missing command line argument for `integration` checks".to_string());
    };

    run_command_with_env(
        "cargo",
        &["install", "--path", ".", "--force", "--locked"],
        ".",
        &HashMap::from([
            ("CFG_RELEASE", "nightly"),
            ("CFG_RELEASE_CHANNEL", "nightly"),
        ]),
    )?;

    println!("Integration tests for {integration}");

    run_command("cargo", &["fmt", "--", "--version"], ".")?;

    match integration.as_str() {
        "cargo" => run_test(
            &integration,
            format!("https://github.com/rust-lang/{integration}.git"),
            HashMap::from([("CFG_DISABLE_CROSS_TESTS", "1")]),
            check_fmt_with_all_tests,
        ),
        "crater" => run_test(
            &integration,
            format!("https://github.com/rust-lang/{integration}.git"),
            HashMap::new(),
            check_fmt_with_lib_tests,
        ),
        "bitflags" => run_test(
            &integration,
            format!("https://github.com/bitflags/{integration}.git"),
            HashMap::new(),
            check_fmt_with_all_tests,
        ),
        "tempdir" => run_test(
            &integration,
            format!("https://github.com/rust-lang-deprecated/{integration}.git"),
            HashMap::new(),
            check_fmt_with_all_tests,
        ),
        _ => run_test(
            &integration,
            format!("https://github.com/rust-lang/{integration}.git"),
            HashMap::new(),
            check_fmt_with_all_tests,
        ),
    }
}
