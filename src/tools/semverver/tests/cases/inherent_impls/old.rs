#[allow(dead_code)]
pub struct Abc { }

#[allow(dead_code)]
impl Abc {
    const A: u8 = 3;

    pub fn abc(&self) -> u8 {
        0
    }

    fn def(&self) { }

    pub fn ghi<A>(&self, a: A) -> A {
        a
    }
}

#[allow(dead_code)]
pub struct Def<A> {
    field: A,
}

#[allow(dead_code)]
impl Def<bool> {
    pub fn def(&self) -> u8 {
        0
    }
}

#[allow(dead_code)]
impl Def<u8> {
    pub fn def(&self) -> u8 {
        0
    }

    pub fn ghi() { }

    fn ghi2() { }
}
