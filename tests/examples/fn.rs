pub mod old {
    pub fn abc() {}

    pub fn bcd() {}

    pub fn cde() {}

    pub fn def(_: u8) {}

    pub fn efg(a: u8, _: u8) -> u8 {
        a
    }

    pub fn fgh(a: u8, _: u8) -> u8 {
        a
    }

    pub fn ghi(a: u8, _: u8) -> u8 {
        a
    }
}

pub mod new {
    pub fn abc() {}

    pub fn bcd(_: u8) {}

    pub fn cde() -> u16 {
        0xcde
    }

    pub fn def() {}

    // TODO: this should be non-breaking (even after we make TypeParameterAdded something
    // that is not `Breaking`)
    pub fn efg<A>(a: A, _: A) -> A {
        a
    }

    pub fn fgh(a: u8, _: u16) -> u8 {
        a
    }

    pub fn ghi(a: u8, _: u8) -> u16 {
        a as u16
    }
}
