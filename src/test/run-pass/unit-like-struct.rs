struct Foo;

fn main() {
    let x: Foo = Foo;
    match x {
        Foo => { io::println("hi"); }
    }
}

