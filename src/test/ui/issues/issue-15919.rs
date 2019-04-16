// error-pattern: too big for the current architecture
// normalize-stderr-test "\[usize; \d+\]" -> "[usize; N]"

// FIXME https://github.com/rust-lang/rust/issues/59774
// normalize-stderr-test "thread.*panicked.*Metadata module not compiled.*\n" -> ""
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""

#[cfg(target_pointer_width = "32")]
fn main() {
    let x = [0usize; 0xffff_ffff];
}

#[cfg(target_pointer_width = "64")]
fn main() {
    let x = [0usize; 0xffff_ffff_ffff_ffff];
}
