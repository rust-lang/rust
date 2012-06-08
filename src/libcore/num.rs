#[doc="An interface for numbers."]

iface num {
    // FIXME: Cross-crate overloading doesn't work yet.
    // FIXME: Interface inheritance.
    fn add(&&other: self) -> self;
    fn sub(&&other: self) -> self;
    fn mul(&&other: self) -> self;
    fn div(&&other: self) -> self;
    fn modulo(&&other: self) -> self;
    fn neg() -> self;

    fn to_int() -> int;
    fn from_int(n: int) -> self;    // TODO: Static functions.
}

