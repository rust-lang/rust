// normalize-stderr-test "\[&usize; \d+\]" -> "[&usize; N]"
// error-pattern: too big for the current architecture

// FIXME https://github.com/rust-lang/rust/issues/59774
// normalize-stderr-test "thread.*panicked.*Metadata module not compiled.*\n" -> ""
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""

#![feature(box_syntax)]

#[cfg(target_pointer_width = "64")]
fn main() {
    let n = 0_usize;
    let a: Box<_> = box [&n; 0xF000000000000000_usize];
    println!("{}", a[0xFFFFFF_usize]);
}

#[cfg(target_pointer_width = "32")]
fn main() {
    let n = 0_usize;
    let a: Box<_> = box [&n; 0xFFFFFFFF_usize];
    println!("{}", a[0xFFFFFF_usize]);
}
