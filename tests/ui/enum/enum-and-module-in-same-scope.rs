enum Foo {
    X
}

mod Foo { //~ ERROR the name `Foo` is defined multiple times
    pub static X: isize = 42;
    fn f() { f() } // Check that this does not result in a resolution error
    //~^ WARN cannot return without recursing
}

fn main() {}
