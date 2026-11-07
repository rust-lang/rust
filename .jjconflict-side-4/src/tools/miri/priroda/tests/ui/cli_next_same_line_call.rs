// Verifies `next` at a same-line callee/caller location.
// Keep the breakpoint line number in the .stdin file in sync with this file.
#[rustfmt::skip]
fn same_line() { fn callee() {} callee(); let after = 1; let _ = after; }

fn main() {
    same_line();
    let after_same_line = 2;
    let _ = after_same_line;
}
