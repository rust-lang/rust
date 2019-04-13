// error-pattern:; 1518600000

// FIXME https://github.com/rust-lang/rust/issues/59774
// normalize-stderr-test "thread.*panicked.*Metadata module not compiled.*\n" -> ""
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""

fn generic<T: Copy>(t: T) {
    let s: [T; 1518600000] = [t; 1518600000];
}

fn main() {
    let x: [u8; 1518599999] = [0; 1518599999];
    generic::<[u8; 1518599999]>(x);
}
