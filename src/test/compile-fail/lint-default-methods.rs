#[forbid(default_methods)];

trait Foo { //~ ERROR default methods are experimental
    fn bar(&self) { println("hi"); }
}

fn main() {}
