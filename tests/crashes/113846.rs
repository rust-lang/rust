//@ known-bug: #113846
trait Www {
    type W;
}

trait Xxx: Www<W = Self::X> {
    type X;
}

trait Yyy: Xxx {}

trait Zzz<'a>: Yyy + Xxx<X = Self::Z> {
    type Z;
}

trait Aaa {
    type Y: Yyy;
}

trait Bbb: Aaa<Y = Self::B> {
    type B: for<'a> Zzz<'a>;
}

impl<T> Bbb for T
where
    T: Aaa,
    T::Y: for<'a> Zzz<'a>,
{
    type B = T::Y;
}

pub fn main() {}
