#[crate_type="lib"];

pub struct Struct {
    x: int
}

impl Struct {
    fn static_meth_struct() -> Struct {
        Struct { x: 1 }
    }

    fn meth_struct(&self) -> int {
        self.x
    }
}

pub enum Enum {
    Variant1(int),
    Variant2(int)
}

impl Enum {
    fn static_meth_enum() -> Enum {
        Variant2(10)
    }

    fn meth_enum(&self) -> int {
        match *self {
            Variant1(x) |
            Variant2(x) => x
        }
    }
}
