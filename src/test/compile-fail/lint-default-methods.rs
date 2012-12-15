#[forbid(default_methods)];

trait Foo {
    fn bar() { io::println("hi"); } //~ ERROR default methods are experimental
}

fn main() {}

