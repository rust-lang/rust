trait Foo {
    fn orange(&self);
    fn orange(&self); //~ ERROR the name `orange` is defined multiple times
}

fn main() {}
