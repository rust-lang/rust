struct Foo(int, int);

fn main() {
    let x = Foo(1, 2);
    match x {   //~ ERROR non-exhaustive
        Foo(1, b) => io::println(fmt!("%d", b)),
        Foo(2, b) => io::println(fmt!("%d", b))
    }
}


