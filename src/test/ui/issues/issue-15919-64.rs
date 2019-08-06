// ignore-32bit

// FIXME https://github.com/rust-lang/rust/issues/59774
// normalize-stderr-test "thread.*panicked.*Metadata module not compiled.*\n" -> ""
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""

fn main() {
    let x = [0usize; 0xffff_ffff_ffff_ffff]; //~ ERROR too big
}
