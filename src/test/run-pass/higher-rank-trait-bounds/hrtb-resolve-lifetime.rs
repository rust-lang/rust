// run-pass
#![allow(dead_code)]
// A basic test of using a higher-ranked trait bound.

// pretty-expanded FIXME #23616

trait FnLike<A,R> {
    fn call(&self, arg: A) -> R;
}

type FnObject<'b> = dyn for<'a> FnLike<&'a isize, &'a isize> + 'b;

fn main() {
}
