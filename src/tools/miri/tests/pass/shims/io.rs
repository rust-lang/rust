use std::io::{self, IsTerminal};

fn main() {
    // We can't really assume that this is truly a terminal, and anyway on Windows Miri will always
    // return `false` here, but we can check that the call succeeds.
    io::stdout().is_terminal();

    // Ensure we can format `io::Error` created from OS errors
    // (calls OS-specific error formatting functions).
    let raw_os_error = if cfg!(unix) {
        22 // EINVAL (on most Unixes, anyway)
    } else if cfg!(windows) {
        87 // ERROR_INVALID_PARAMETER
    } else {
        panic!("unsupported OS")
    };
    let err = io::Error::from_raw_os_error(raw_os_error);
    let _ = format!("{err}: {err:?}");
}
