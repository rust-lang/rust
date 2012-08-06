// Testing that the B's are resolved

trait clam<A> { }

enum foo = int;

impl foo {
    fn bar<B,C:clam<B>>(c: C) -> B { fail; }
}

fn main() { }