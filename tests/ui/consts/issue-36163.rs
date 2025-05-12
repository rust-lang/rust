const A: isize = Foo::B as isize;

enum Foo {
    B = A, //~ ERROR E0391
}

fn main() {}
