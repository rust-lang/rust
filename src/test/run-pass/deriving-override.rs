trait MyEq {
    #[derivable]
    pure fn eq(&self, other: &self) -> bool;
    #[derivable]
    pure fn ne(&self, other: &self) -> bool;
}

struct A {
    x: int
}

impl int : MyEq {
    pure fn eq(&self, other: &int) -> bool { *self == *other }
    pure fn ne(&self, other: &int) -> bool { *self != *other }
}

impl A : MyEq {
    pure fn ne(&self, other: &A) -> bool { !self.eq(other) }
}

fn main() {
    let a = A { x: 1 };
    assert a.eq(&a);
    assert !a.ne(&a);
}

