//! An interface for numeric types

trait Num {
    // FIXME: Trait composition. (#2616)
    pure fn add(&&other: self) -> self;
    pure fn sub(&&other: self) -> self;
    pure fn mul(&&other: self) -> self;
    pure fn div(&&other: self) -> self;
    pure fn modulo(&&other: self) -> self;
    pure fn neg() -> self;

    pure fn to_int() -> int;
    static pure fn from_int(n: int) -> self;
}
