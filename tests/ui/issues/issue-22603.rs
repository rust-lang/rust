// check-pass

#![feature(unboxed_closures, fn_traits)]

struct Foo;

impl<A> FnOnce<(A,)> for Foo {
    type Output = ();
    extern "rust-call" fn call_once(self, (_,): (A,)) {}
}

impl<A> std::ops::Callable<(A,)> for Foo {}

fn main() {
    println!("{:?}", Foo("bar"));
}
