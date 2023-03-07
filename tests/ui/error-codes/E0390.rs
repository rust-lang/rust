struct Foo {
    x: i32
}

impl *mut Foo {} //~ ERROR E0390

impl fn(Foo) {} //~ ERROR E0390

fn main() {
}
