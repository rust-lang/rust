// compile-flags: -Zunstable-options
#![stable(feature = "stable", since = "1.0.0")]
#![feature(staged_api, rustc_attrs)]

#[unstable(feature = "unstable", issue = "none")]
#[rustc_const_stable(feature = "stable", since = "1.0.0")]
const fn foo() {} //~ ERROR functions cannot be const-stable if they are unstable

mod bar {
    #![unstable(feature = "unstable", issue = "none")]

    #[rustc_const_stable(feature = "stable", since = "1.0.0")]
    const fn foo() {} //~ ERROR functions cannot be const-stable if they are unstable
}

fn main() {}
