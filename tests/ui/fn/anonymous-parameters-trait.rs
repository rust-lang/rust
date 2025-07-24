//@ edition: 2015
//@ check-pass

trait Foo {
    #[allow(anonymous_parameters)]
    fn quux(u8) {}
}

fn main() {}
