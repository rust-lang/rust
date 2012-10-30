struct Foo(int, int);

fn main() {
    let x = Foo(1, 2);
    let Foo(y, z) = x;
    io::println(fmt!("%d %d", y, z));
    assert y == 1;
    assert z == 2;
}

