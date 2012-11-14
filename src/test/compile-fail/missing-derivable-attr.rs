trait MyEq {
    pure fn eq(other: &self) -> bool;
}

struct A {
    x: int
}

impl int : MyEq {
    pure fn eq(other: &int) -> bool { self == *other }
}

impl A : MyEq;  //~ ERROR missing method

fn main() {
}

