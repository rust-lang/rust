/// An interface for numbers.

iface num {
    // FIXME: Cross-crate overloading doesn't work yet. (#2615)
    // FIXME: Interface inheritance. (#2616)
    fn add(&&other: self) -> self;
    fn sub(&&other: self) -> self;
    fn mul(&&other: self) -> self;
    fn div(&&other: self) -> self;
    fn modulo(&&other: self) -> self;
    fn neg() -> self;

    fn to_int() -> int;
    fn from_int(n: int) -> self;    // FIXME (#2376) Static functions.
    // n.b. #2376 is for classes, not ifaces, but it could be generalized...
}

