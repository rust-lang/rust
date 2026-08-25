//@ check-pass

#![deny(dead_code)]

trait UInt: Copy + From<u8> {}

impl UInt for u16 {}

trait Int: Copy {
    type Unsigned: UInt;

    fn as_unsigned(self) -> Self::Unsigned;
}

impl Int for i16 {
    type Unsigned = u16;

    fn as_unsigned(self) -> u16 {
        self as _
    }
}

fn priv_func<T: Int>(x: u8, y: T) -> (T::Unsigned, T::Unsigned) {
    (T::Unsigned::from(x), y.as_unsigned())
}

pub fn pub_func(x: u8, y: i16) -> (u16, u16) {
    priv_func(x, y)
}

fn main() {}
