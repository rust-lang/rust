// run-pass
// pretty-expanded FIXME #23616

trait MyTrait {
    fn foo(&self);
}

impl<A, B, C> MyTrait for fn(A, B) -> C {
    fn foo(&self) {}
}

fn bar<T: MyTrait>(t: &T) {
    t.foo()
}

fn thing(a: isize, b: isize) -> isize {
    a + b
}

fn main() {
    let thing: fn(isize, isize) -> isize = thing; // coerce to fn type
    bar(&thing);
}
