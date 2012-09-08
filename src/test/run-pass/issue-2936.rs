trait bar<T> {
    fn get_bar() -> T;
}

fn foo<T, U: bar<T>>(b: U) -> T {
    b.get_bar()
}

struct cbar {
    x: int,
}

impl cbar : bar<int> {
    fn get_bar() -> int {
        self.x
    }
}

fn cbar(x: int) -> cbar {
    cbar {
        x: x
    }
}

fn main() {
    let x: int = foo::<int, cbar>(cbar(5));
    assert x == 5;
}
