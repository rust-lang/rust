//@ revisions: unoptimized optimized
//@[optimized]compile-flags: -O
//@ run-pass
//@ only-64bit
//@ min-llvm-version: 22

// Regression test for https://github.com/rust-lang/rust/issues/83060

fn func() {
    const CAP: usize = std::u32::MAX as usize;
    let mut x: [u8; CAP] = [0; CAP];
    x[2] = 123;
    assert_eq!(x[2], 123);
}

fn main() {
    std::thread::Builder::new()
        .stack_size(5 * 1024 * 1024 * 1024)
        .spawn(func)
        .unwrap()
        .join()
        .unwrap();
}
