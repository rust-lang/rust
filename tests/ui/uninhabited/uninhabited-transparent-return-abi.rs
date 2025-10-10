//@ run-pass
//@ needs-unwind
//@ ignore-backends: gcc
// See https://github.com/rust-lang/rust/issues/135802

enum Void {}

// Should be ABI-compatible with T, but wasn't prior to the PR adding this test.
#[repr(transparent)]
struct NoReturn<T>(T, Void);

// Returned by invisible reference (in most ABIs)
#[allow(dead_code)]
struct Large(u64, u64, u64);

// Prior to the PR adding this test, this function had a different ABI than
// `fn() -> Large` (on `x86_64-unknown-linux-gnu` at least), so calling it as `fn() -> Large`
// would pass the return place pointer in rdi and `correct` in rsi, but the function
// would expect `correct` in rdi.
fn never(correct: &mut bool) -> NoReturn<Large> {
    *correct = true;
    panic!("catch this")
}

fn main() {
    let mut correct = false;
    let never: fn(&mut bool) -> NoReturn<Large> = never;
    // Safety: `NoReturn<Large>` is a `repr(transparent)` wrapper around `Large`,
    // so they should be ABI-compatible.
    let never: fn(&mut bool) -> Large = unsafe { std::mem::transmute(never) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| never(&mut correct)));
    assert!(result.is_err(), "function should have panicked");
    assert!(correct, "function should have stored `true` into `correct`");
}
