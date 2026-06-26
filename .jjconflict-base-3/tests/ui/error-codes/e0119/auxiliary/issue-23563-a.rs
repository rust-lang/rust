// Ref: https://github.com/rust-lang/rust/issues/23563#issuecomment-260751672

pub trait LolTo<T> {
    fn convert_to(&self) -> T;
}

pub trait LolInto<T>: Sized {
    fn convert_into(self) -> T;
}

pub trait LolFrom<T> {
    fn from(_: T) -> Self;
}

impl<'a, T: ?Sized, U> LolInto<U> for &'a T where T: LolTo<U> {
    fn convert_into(self) -> U {
        self.convert_to()
    }
}

impl<T, U> LolFrom<T> for U where T: LolInto<U> {
    fn from(t: T) -> U {
        t.convert_into()
    }
}
