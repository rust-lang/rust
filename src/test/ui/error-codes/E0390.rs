struct Foo {
    x: i32
}

impl *mut Foo {} //~ ERROR E0390

fn main() {
}
