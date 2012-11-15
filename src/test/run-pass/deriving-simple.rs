trait MyEq {
    #[derivable]
    pure fn eq(&self, other: &self) -> bool;
}

struct A {
    x: int
}

struct B {
    x: A,
    y: A,
    z: A
}

impl A : MyEq {
    pure fn eq(&self, other: &A) -> bool {
        unsafe { io::println(fmt!("eq %d %d", self.x, other.x)); }
        self.x == other.x
    }
}

impl B : MyEq;

fn main() {
    let b = B { x: A { x: 1 }, y: A { x: 2 }, z: A { x: 3 } };
    let c = B { x: A { x: 1 }, y: A { x: 3 }, z: A { x: 4 } };
    assert b.eq(&b);
    assert c.eq(&c);
    assert !b.eq(&c);
    assert !c.eq(&b);
}

