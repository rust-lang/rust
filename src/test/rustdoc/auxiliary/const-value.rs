use std::cell::Cell;

pub struct Data {
    pub open: (i8, i8, i8),
    closed: bool,
    #[doc(hidden)]
    pub internal: Cell<u64>,
}

impl Data {
    pub const fn new(value: (i8, i8, i8)) -> Self {
        Self {
            open: value,
            closed: false,
            internal: Cell::new(0),
        }
    }
}

pub struct Opaque(u32);

impl Opaque {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
}
