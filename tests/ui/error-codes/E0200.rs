struct Foo;

unsafe trait Bar { }

impl Bar for Foo { } //~ ERROR E0200

fn main() {
}
