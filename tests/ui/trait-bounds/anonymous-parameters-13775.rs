//@ edition: 2015
//@ check-pass

trait Foo {
    #[allow(anonymous_parameters)]
    fn bar(&self, isize) {}
}

fn main() {}
