trait MyEq {
    pure fn eq(other: &self) -> bool;
}

impl int : MyEq {
    pure fn eq(other: &int) -> bool {
        self == *other
    }
}

impl<T:MyEq> @T : MyEq {
    pure fn eq(other: &@T) -> bool {
        unsafe {
            io::println("@T");
        }
        (*self).eq(&**other)
    }
}

struct A {
    x: @int,
    y: @int
}

impl A : MyEq;

fn main() {
    let a = A { x: @3, y: @5 };
    let b = A { x: @10, y: @20 };
    assert a.eq(&a);
    assert !a.eq(&b);
}

