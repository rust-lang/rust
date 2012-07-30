trait Product {
    fn product() -> int;
}

struct Foo {
    x: int;
    y: int;
}

impl Foo {
    fn sum() -> int {
        self.x + self.y
    }
}

impl Foo : Product {
    fn product() -> int {
        self.x * self.y
    }
}

fn Foo(x: int, y: int) -> Foo {
    Foo { x: x, y: y }
}

fn main() {
    let foo = Foo(3, 20);
    io::println(fmt!{"%d %d", foo.sum(), foo.product()});
}

