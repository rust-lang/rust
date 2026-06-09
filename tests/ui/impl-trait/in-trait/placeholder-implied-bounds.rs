//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

pub fn main() {}

pub trait Iced {
    fn get(&self) -> &impl Sized;
}

/// Impl causes ICE
impl Iced for () {
    fn get(&self) -> &impl Sized {
        &()
    }
}
