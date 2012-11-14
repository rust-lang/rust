trait MyEq {
    #[derivable]
    pure fn eq(other: &self) -> bool;
    #[derivable]
    pure fn ne(other: &self) -> bool;
}

struct A {
    x: int
}

impl int : MyEq {
    pure fn eq(other: &int) -> bool { self == *other }
    pure fn ne(other: &int) -> bool { self != *other }
}

impl A : MyEq {
    pure fn ne(other: &A) -> bool { !self.eq(other) }
}

fn main() {
    let a = A { x: 1 };
    assert a.eq(&a);
    assert !a.ne(&a);
}

