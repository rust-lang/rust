// Verifies duplicate MIR locations on one source breakpoint line are skipped.
// The following line gives the second `continue` a later real breakpoint.
// This keeps the test from passing merely because the program finished.
// Keep the breakpoint line numbers in the .stdin file in sync with this file.
fn main() {
    let _first = 0;
    let _second = 1;
}
