#[must_use]
pub struct S;

pub struct T;

pub struct B {
    s: S,
    pub t: T,
}

pub struct C {
    pub s: S,
    pub t: T,
}

impl B {
    pub fn new() -> B {
        B { s: S, t: T }
    }
}

impl C {
    pub fn new() -> C {
        C { s: S, t: T }
    }
}
