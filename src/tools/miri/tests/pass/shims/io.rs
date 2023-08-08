use std::io::IsTerminal;

fn main() {
    // We can't really assume that this is truly a terminal, and anyway on Windows Miri will always
    // return `false` here, but we can check that the call succeeds.
    std::io::stdout().is_terminal();
}
