struct Foo(int, int);

fn main() {
    let x = Foo(1, 2);
    match x {
        Foo(a, b) => {
            assert a == 1;
            assert b == 2;
            io::println(fmt!("%d %d", a, b));
        }
    }
}

