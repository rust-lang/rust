pub struct Named {
    hidden: u8,
}

impl Named {
    pub fn new() -> Self {
        Self { hidden: 0 }
    }
}

pub struct NamedWithMultipleFields {
    hidden: u8,
    pub visible: u8,
}

struct PrivateInner;

pub struct PublicTuple(PrivateInner);

impl PublicTuple {
    pub fn new() -> Self {
        Self(PrivateInner)
    }
}
