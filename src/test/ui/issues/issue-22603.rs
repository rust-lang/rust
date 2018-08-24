#![feature(unboxed_closures, fn_traits, rustc_attrs)]

struct Foo;

impl<A> FnOnce<(A,)> for Foo {
    type Output = ();
    extern "rust-call" fn call_once(self, (_,): (A,)) {
    }
}
#[rustc_error]
fn main() { //~ ERROR compilation successful
    println!("{:?}", Foo("bar"));
}
