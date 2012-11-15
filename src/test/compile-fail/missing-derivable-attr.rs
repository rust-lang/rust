trait MyEq {
    pure fn eq(&self, other: &self) -> bool;
}

struct A {
    x: int
}

impl int : MyEq {
    pure fn eq(&self, other: &int) -> bool { *self == *other }
}

impl A : MyEq;  //~ ERROR missing method

fn main() {
}

