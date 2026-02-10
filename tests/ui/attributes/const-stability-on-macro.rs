#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

#[rustc_const_stable(feature = "foo", since = "3.3.3")]
//~^ ERROR attribute cannot be used on macro defs
macro_rules! foo {
    () => {};
}

#[rustc_const_unstable(feature = "bar", issue = "none")]
//~^ ERROR attribute cannot be used on macro defs
macro_rules! bar {
    () => {};
}

fn main() {}
