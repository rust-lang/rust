#![unstable(feature = "humans",
            reason = "who ever let humans program computers,
            we're apparently really bad at it",
            issue = "0")]

#![feature(rustc_const_unstable, const_fn, foo, foo2)]
#![feature(staged_api)]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature="foo")]
const fn foo() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const fn bar() -> u32 { foo() } //~ ERROR can only call other `const fn`

#[unstable(feature = "rust1", issue="0")]
const fn foo2() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const fn bar2() -> u32 { foo2() } //~ ERROR can only call other `const fn`

#[stable(feature = "rust1", since = "1.0.0")]
// conformity is required, even with `const_fn` feature gate
const fn bar3() -> u32 { (5f32 + 6f32) as u32 } //~ ERROR only int, `bool` and `char` operations

// check whether this function cannot be called even with the feature gate active
#[unstable(feature = "foo2", issue="0")]
const fn foo2_gated() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const fn bar2_gated() -> u32 { foo2_gated() } //~ ERROR can only call other `const fn`

fn main() {}
