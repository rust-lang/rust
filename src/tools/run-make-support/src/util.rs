use crate::command::{Command, CompletedProcess};

/// If a given [`Command`] failed (as indicated by its [`CompletedProcess`]), verbose print the
/// executed command, failure location, output status and stdout/stderr, and abort the process with
/// exit code `1`.
pub(crate) fn handle_failed_output(
    cmd: &Command,
    output: CompletedProcess,
    caller_line_number: u32,
) -> ! {
    if output.status().success() {
        eprintln!("command unexpectedly succeeded at line {caller_line_number}");
    } else {
        eprintln!("command failed at line {caller_line_number}");
    }
    eprintln!("{cmd:?}");
    eprintln!("output status: `{}`", output.status());
    eprintln!("=== STDOUT ===\n{}\n\n", output.stdout_utf8());
    eprintln!("=== STDERR ===\n{}\n\n", output.stderr_utf8());
    std::process::exit(1)
}
