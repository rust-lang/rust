// compile-flags: -Zunstable-options
#![stable(feature = "stable", since = "1.0.0")]
#![feature(staged_api, rustc_attrs, intrinsics)]

#[unstable(feature = "unstable", issue = "none")]
#[rustc_const_stable(feature = "stable", since = "1.0.0")]
const fn foo() {} //~ ERROR functions cannot be const-stable if they are unstable

mod bar {
    #![unstable(feature = "unstable", issue = "none")]

    #[rustc_const_stable(feature = "stable", since = "1.0.0")]
    const fn foo() {} //~ ERROR functions cannot be const-stable if they are unstable
}

mod intrinsics {
    #![unstable(feature = "unstable", issue = "none")]
    extern "rust-intrinsic" {
        #[rustc_const_stable(feature = "stable", since = "1.0.0")]
        pub fn transmute<T, U>(_: T) -> U; //~ ERROR functions cannot be const-stable if they are unstable
    }
}

fn main() {}
