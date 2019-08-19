// check-pass

#![feature(unboxed_closures, fn_traits)]

struct Foo;

impl<A> FnOnce<(A,)> for Foo {
    type Output = ();
    extern "rust-call" fn call_once(self, (_,): (A,)) {
    }
}

fn main() {
    println!("{:?}", Foo("bar"));
}
