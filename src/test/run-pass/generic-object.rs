trait Foo<T> {
    fn get() -> T;
}

struct S {
    x: int
}

impl S : Foo<int> {
    fn get() -> int {
        self.x
    }
}

fn main() {
    let x = @S { x: 1 };
    let y = x as @Foo<int>;
    assert y.get() == 1;
}

