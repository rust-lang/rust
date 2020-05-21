#![stable(feature = "rust1", since = "1.0.0")]

#![feature(staged_api)]
#![feature(const_loop, const_fn)]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
const fn foo() -> i32 {
    loop { return 42; } //~ ERROR `loop` is not allowed in a `const fn`
}

fn main() {}
