trait MyEq {
    #[derivable]
    pure fn eq(other: &self) -> bool;
}

struct A {
    x: int
}

enum B {
    C(A),
    D(A),
    E(A)
}

impl B : MyEq;
//~^ ERROR cannot automatically derive
//~^^ ERROR cannot automatically derive
//~^^^ ERROR cannot automatically derive

fn main() {
    let c = C(A { x: 15 });
    assert c.eq(&c);
}

