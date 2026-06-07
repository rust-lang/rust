use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Runs one scripted Priroda CLI fixture against a Rust program.
///
/// `test_path` is the extensionless fixture stem. The helper sends
/// `<test_path>.stdin` to Priroda and expects exact stdout from
/// `<test_path>.stdout`.
fn run_cli_test(program_path: &str, test_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Anchor relative paths to the crate root so the test does not depend on
    // the current working directory used by a test runner or IDE.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let test_path = manifest_dir.join(test_path);
    let stdin_path = test_path.with_extension("stdin");
    let stdout_path = test_path.with_extension("stdout");

    // Keep scripted stdin as raw bytes: the helper only forwards it to the
    // child process, so there is no need to validate it as UTF-8 here.
    let input = std::fs::read(stdin_path)?;

    // Source paths printed for `std` frames come from the active rustc
    // toolchain, not from `MIRI_SYSROOT`, so expand `{RUSTC_SYSROOT}` from
    // `rustc --print sysroot`. Use strict UTF-8 because this value is part of
    // the exact stdout contract; invalid output should fail the test.
    let rustc_sysroot = Command::new("rustc").arg("--print").arg("sysroot").output()?;
    let rustc_sysroot = String::from_utf8(rustc_sysroot.stdout)?.trim().to_owned();

    // Keep fixture stdout exact while allowing machine-specific absolute paths
    // to be written with stable placeholders.
    let expected_output = std::fs::read_to_string(stdout_path)?
        .replace("{MANIFEST_DIR}", &manifest_dir.display().to_string())
        .replace("{MIRI_DIR}", &manifest_dir.parent().unwrap().display().to_string())
        .replace("{RUSTC_SYSROOT}", &rustc_sysroot);

    let mut priroda = Command::new(env!("CARGO_BIN_EXE_priroda"))
        .arg(manifest_dir.join(program_path))
        // The CLI contract checks stderr exactly, so inherited logging
        // configuration would make the test fail for reasons unrelated to
        // Priroda's behavior.
        .env_remove("RUSTC_LOG")
        .env_remove("RUST_LOG")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    priroda.stdin.as_mut().unwrap().write_all(&input)?;
    // Close stdin after the scripted input so Priroda can observe EOF if a
    // fixture does not explicitly quit.
    drop(priroda.stdin.take());

    let output = priroda.wait_with_output()?;

    assert!(
        output.status.success(),
        "priroda exited with status {}\nstderr:\n{}",
        output.status,
        // This is only diagnostic text for a failed assertion, so lossy UTF-8 is
        // better than hiding stderr behind a second conversion failure.
        String::from_utf8_lossy(&output.stderr),
    );

    assert!(
        output.stderr.is_empty(),
        "expected no stderr output, got:\n{}",
        // Same reasoning as above: stderr is not part of the success path here;
        // it is shown to make a failure easier to debug.
        String::from_utf8_lossy(&output.stderr),
    );

    // Actual stdout is the value under test, so require valid UTF-8 before doing
    // the exact string comparison against the expanded fixture.
    assert_eq!(String::from_utf8(output.stdout)?, expected_output);

    Ok(())
}

/// Verifies Priroda can start on the simplest passing Rust program and accept
/// a scripted `quit` command.
#[test]
fn empty_main() -> Result<(), Box<dyn std::error::Error>> {
    run_cli_test("../tests/pass/empty_main.rs", "tests/cli/empty_main")
}

/// Verifies EOF exits the debugger loop cleanly without requiring an explicit
/// quit command.
#[test]
fn eof_exits_cleanly() -> Result<(), Box<dyn std::error::Error>> {
    run_cli_test("../tests/pass/empty_main.rs", "tests/cli/eof_exits_cleanly")
}

/// Verifies unknown commands and malformed breakpoints are rejected without
/// mutating debugger state.
#[test]
fn invalid_commands() -> Result<(), Box<dyn std::error::Error>> {
    run_cli_test("../tests/pass/empty_main.rs", "tests/cli/invalid_commands")
}

/// Verifies breakpoint aliases and duplicate detection before execution starts.
#[test]
fn duplicate_breakpoint() -> Result<(), Box<dyn std::error::Error>> {
    run_cli_test("../tests/pass/empty_main.rs", "tests/cli/duplicate_breakpoint")
}

/// Verifies continue can drive the interpreted program to normal completion.
#[test]
fn continue_finishes_program() -> Result<(), Box<dyn std::error::Error>> {
    run_cli_test("../tests/pass/empty_main.rs", "tests/cli/continue_finishes_program")
}

/// Verifies continue stops when execution reaches a registered source-location
/// breakpoint.
#[test]
fn continue_hits_breakpoint() -> Result<(), Box<dyn std::error::Error>> {
    run_cli_test("../tests/pass/empty_main.rs", "tests/cli/continue_hits_breakpoint")
}

/// Verifies every current spelling of MIR-instruction stepping advances
/// execution and reports a location.
#[test]
fn step_aliases() -> Result<(), Box<dyn std::error::Error>> {
    run_cli_test("../tests/pass/empty_main.rs", "tests/cli/step_aliases")
}

/// Documents the current repeated-stop behavior when multiple MIR locations map
/// to the same source breakpoint line.
#[test]
fn repeated_same_line_breakpoint() -> Result<(), Box<dyn std::error::Error>> {
    run_cli_test("../tests/pass/empty_main.rs", "tests/cli/repeated_same_line_breakpoint")
}
