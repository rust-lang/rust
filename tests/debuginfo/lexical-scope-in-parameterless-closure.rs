//@ compile-flags:-C debuginfo=1

// gdb-command:run
// lldb-command:run

// Nothing to do here really, just make sure it compiles. See issue #8513.
fn main() {
    let _ = ||();
    let _ = (1_usize..3).map(|_| 5);
}
