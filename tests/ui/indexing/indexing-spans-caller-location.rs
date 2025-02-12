//@ run-pass

// Regression test for https://github.com/rust-lang/rust/issues/114388

#[track_caller]
fn caller_line() -> u32 {
    std::panic::Location::caller().line()
}

fn main() {
    let prev_line = caller_line(); // first line
    (A { prev_line }) // second line
    [0]; // third line
}

struct A {
    prev_line: u32,
}
impl std::ops::Index<usize> for A {
    type Output = ();

    fn index(&self, _idx: usize) -> &() {
        // Use the relative number to make it resistant to header changes.
        assert_eq!(caller_line(), self.prev_line + 2);
        &()
    }
}
