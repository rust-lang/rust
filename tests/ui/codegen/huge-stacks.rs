//@ revisions: unoptimized optimized
//@[optimized]compile-flags: -O
//@ run-pass
//@ only-64bit
//@ min-llvm-version: 22

// Regression test for https://github.com/rust-lang/rust/issues/83060
// Verifies a program is not miscompiled if it includes a 4GB array on the stack

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
        .expect("huge-stacks.rs requires 5GB RAM to run")
        .join()
        .unwrap();
}
