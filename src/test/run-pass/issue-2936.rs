trait bar<T> {
    fn get_bar() -> T;
}

fn foo<T, U: bar<T>>(b: U) -> T {
    b.get_bar()
}

struct cbar : bar<int> {
    x: int;
    new(x: int) { self.x = x; }
    fn get_bar() -> int {
        self.x
    }
}

fn main() {
    let x: int = foo::<int, cbar>(cbar(5));
    assert x == 5;
}
