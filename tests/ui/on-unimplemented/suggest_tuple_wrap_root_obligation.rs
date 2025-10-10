struct Tuple; //~ HELP the trait `From<u8>` is not implemented for `Tuple`

impl From<(u8,)> for Tuple {
    fn from(_: (u8,)) -> Self {
        todo!()
    }
}
impl From<(u8, u8)> for Tuple {
    fn from(_: (u8, u8)) -> Self {
        todo!()
    }
}
impl From<(u8, u8, u8)> for Tuple {
    fn from(_: (u8, u8, u8)) -> Self {
        todo!()
    }
}

fn convert_into_tuple(_x: impl Into<Tuple>) {}

fn main() {
    convert_into_tuple(42_u8);
    //~^ ERROR E0277
    //~| HELP use a unary tuple instead
    //~| HELP the following other types implement trait `From<T>`
}
