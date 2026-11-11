#![crate_type="lib"]

pub struct Struct {
    pub x: isize
}

impl Struct {
    fn static_meth_struct() -> Struct {
        Struct { x: 1 }
    }

    fn meth_struct(&self) -> isize {
        self.x
    }
}

pub enum Enum {
    Variant1(isize),
    Variant2(isize)
}

impl Enum {
    fn static_meth_enum() -> Enum {
        Enum::Variant2(10)
    }

    fn meth_enum(&self) -> isize {
        match *self {
            Enum::Variant1(x) |
            Enum::Variant2(x) => x
        }
    }
}
