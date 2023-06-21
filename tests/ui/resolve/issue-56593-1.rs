// check-pass

struct Foo;

mod foo {
    use super::*;

    #[derive(Debug)]
    pub struct Foo;
}

mod bar {
    use super::foo::*;

    fn bar(_: Foo) {}
}

fn main() {}
