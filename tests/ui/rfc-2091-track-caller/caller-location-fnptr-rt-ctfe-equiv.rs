// Ensure that a `#[track_caller]` function, returning `caller_location()`,
// which coerced (to a function pointer) and called, inside a `const fn`,
// in turn called, results in the same output irrespective of whether
// we're in a const or runtime context.

// run-pass
// compile-flags: -Z unleash-the-miri-inside-of-you

#![feature(core_intrinsics, const_caller_location)]

type L = &'static std::panic::Location<'static>;

#[track_caller]
const fn attributed() -> L {
    std::intrinsics::caller_location()
}

const fn calling_attributed() -> L {
    // We need `-Z unleash-the-miri-inside-of-you` for this as we don't have `const fn` pointers.
    let ptr: fn() -> L = attributed;
    ptr()
}

fn main() {
    const CONSTANT: L = calling_attributed();
    let runtime = calling_attributed();

    assert_eq!(
        (runtime.file(), runtime.line(), runtime.column()),
        (CONSTANT.file(), CONSTANT.line(), CONSTANT.column()),
    );
}
