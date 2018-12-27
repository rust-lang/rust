// compile-pass
pub trait Reader {}

enum Wrapper<'a> {
    WrapReader(&'a (Reader + 'a))
}

trait Wrap<'a> {
    fn wrap(self) -> Wrapper<'a>;
}

impl<'a, R: Reader> Wrap<'a> for &'a mut R {
    fn wrap(self) -> Wrapper<'a> {
        Wrapper::WrapReader(self as &'a mut Reader)
    }
}

pub fn main() {}
