trait MyTrait {
    fn f(&self) -> Self;
}

struct S {
    x: int
}

impl S : MyTrait {
    fn f(&self) -> S {
        S { x: 3 }
    }
}

fn main() {}

