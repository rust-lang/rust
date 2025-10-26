//! Regression test for https://github.com/rust-lang/rust/issues/14901

//@ check-pass
pub trait Reader {}

enum Wrapper<'a> {
    WrapReader(&'a (dyn Reader + 'a))
}

trait Wrap<'a> {
    fn wrap(self) -> Wrapper<'a>;
}

impl<'a, R: Reader> Wrap<'a> for &'a mut R {
    fn wrap(self) -> Wrapper<'a> {
        Wrapper::WrapReader(self as &'a mut dyn Reader)
    }
}

pub fn main() {}
