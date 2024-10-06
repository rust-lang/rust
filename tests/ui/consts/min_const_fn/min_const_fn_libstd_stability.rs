//@ compile-flags: --edition 2018
#![unstable(feature = "humans",
            reason = "who ever let humans program computers,
            we're apparently really bad at it",
            issue = "none")]

#![feature(foo, foo2)]
#![feature(const_async_blocks, staged_api, rustc_attrs)]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature="foo", issue = "none")]
const fn foo() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const fn bar() -> u32 { foo() } //~ ERROR cannot use `#[feature(foo)]`

#[unstable(feature = "foo2", issue = "none")]
#[rustc_const_unstable(feature = "foo2", issue = "none")]
const fn foo2() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const fn bar2() -> u32 { foo2() } //~ ERROR cannot use `#[feature(foo2)]`

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// conformity is required
const fn bar3() -> u32 {
    let x = async { 13 };
    //~^ ERROR cannot use `#[feature(const_async_blocks)]`
    foo()
    //~^ ERROR cannot use `#[feature(foo)]`
}

// check whether this function cannot be called even with the feature gate active
#[unstable(feature = "foo2", issue = "none")]
#[rustc_const_unstable(feature = "foo2", issue = "none")]
const fn foo2_gated() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const fn bar2_gated() -> u32 { foo2_gated() } //~ ERROR cannot use `#[feature(foo2)]`

// Functions with the `rustc_const_stable_indirect` attribute
// are checked like stable functions.
#[rustc_const_stable_indirect]
const fn bar2_gated_stable_indirect() -> u32 { foo2_gated() } //~ ERROR cannot use `#[feature(foo2)]`

// These functiuons *can* be called from fully stable functions.
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
const fn bar2_gated_exposed() -> u32 {
    bar2_gated_stable_indirect()
}

fn main() {}
