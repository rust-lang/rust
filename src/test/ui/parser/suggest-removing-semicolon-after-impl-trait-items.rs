// run-rustfix

trait Foo {
    fn bar() {}; //~ ERROR non-item in item list
}

fn main() {}
