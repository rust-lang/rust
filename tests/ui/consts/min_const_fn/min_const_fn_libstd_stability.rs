#![unstable(feature = "humans",
            reason = "who ever let humans program computers,
            we're apparently really bad at it",
            issue = "none")]

#![feature(const_refs_to_cell, foo, foo2)]
#![feature(staged_api)]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature="foo", issue = "none")]
const fn foo() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const fn bar() -> u32 { foo() } //~ ERROR not yet stable as a const fn

#[unstable(feature = "foo2", issue = "none")]
const fn foo2() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const fn bar2() -> u32 { foo2() } //~ ERROR not yet stable as a const fn

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// conformity is required
const fn bar3() -> u32 {
    let x = std::cell::Cell::new(0u32);
    x.get();
    //~^ ERROR const-stable function cannot use `#[feature(const_refs_to_cell)]`
    //~| ERROR cannot call non-const fn
    foo()
    //~^ ERROR is not yet stable as a const fn
}

// check whether this function cannot be called even with the feature gate active
#[unstable(feature = "foo2", issue = "none")]
const fn foo2_gated() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const fn bar2_gated() -> u32 { foo2_gated() } //~ ERROR not yet stable as a const fn

fn main() {}
