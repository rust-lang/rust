#![unstable(feature = "humans",
            reason = "who ever let humans program computers,
            we're apparently really bad at it",
            issue = "none")]

#![feature(foo, foo2)]
#![feature(staged_api)]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature="foo", issue = "none")]
const unsafe fn foo() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const unsafe fn bar() -> u32 { unsafe { foo() } } //~ ERROR cannot use `#[feature(foo)]`

#[unstable(feature = "foo2", issue = "none")]
#[rustc_const_unstable(feature = "foo2", issue = "none")]
const unsafe fn foo2() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const unsafe fn bar2() -> u32 { unsafe { foo2() } } //~ ERROR cannot use `#[feature(foo2)]`

// check whether this function cannot be called even with the feature gate active
#[unstable(feature = "foo2", issue = "none")]
#[rustc_const_unstable(feature = "foo2", issue = "none")]
const unsafe fn foo2_gated() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const unsafe fn bar2_gated() -> u32 { unsafe { foo2_gated() } }
//~^ ERROR cannot use `#[feature(foo2)]`

fn main() {}
