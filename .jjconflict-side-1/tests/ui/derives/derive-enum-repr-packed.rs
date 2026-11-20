//! Regression test for https://github.com/rust-lang/rust/issues/133025.

#[derive(Debug)]
#[repr(packed)] //~ ERROR: the `repr(packed)` attribute cannot be used on enums
enum COption<T> {
    None,
    Some(T),
}

fn main() {}
