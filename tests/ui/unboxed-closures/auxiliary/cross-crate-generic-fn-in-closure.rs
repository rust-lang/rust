//! Auxiliary crate for <https://github.com/rust-lang/rust/issues/18711>.

#![crate_type = "rlib"]

pub fn inner<F>(f: F) -> F {
    (move || f)()
}
