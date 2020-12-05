// run-rustfix

trait Foo {
    fn foo([a, b]: [i32; 2]) {}
    //~^ ERROR: patterns aren't allowed in methods without bodies
}

fn main() {}
