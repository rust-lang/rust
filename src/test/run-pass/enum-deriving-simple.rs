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

impl A : MyEq {
    pure fn eq(other: &A) -> bool {
        unsafe { io::println("in eq"); }
        self.x == other.x
    }
}

impl B : MyEq;

fn main() {
    let c = C(A { x: 15 });
    let d = D(A { x: 30 });
    let e = C(A { x: 30 });
    assert c.eq(&c);
    assert !c.eq(&d);
    assert !c.eq(&e);
}

