struct Foo(int, int);

fn main() {
    let x = Foo(1, 2);
    io::println(fmt!("%?", x));
}

