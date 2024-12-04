trait EnumSetType {
    type Repr;
}

enum Enum8 { }
impl EnumSetType for Enum8 {
    type Repr = u8;
}

#[derive(PartialEq, Eq)]
struct EnumSet<T: EnumSetType> {
    __enumset_underlying: T::Repr,
}

const CONST_SET: EnumSet<Enum8> = EnumSet { __enumset_underlying: 3 };

fn main() {
    match CONST_SET {
        CONST_SET => { /* ok */ } //~ ERROR constant of non-structural type `EnumSet<Enum8>` in a pattern
        _ => panic!("match fell through?"),
    }
}
