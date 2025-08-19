//! Regression test for https://github.com/rust-lang/rust/issues/12127

#![feature(unboxed_closures, tuple_trait)]

fn to_fn_once<A:std::marker::Tuple,F:FnOnce<A>>(f: F) -> F { f }
fn do_it(x: &isize) { }

fn main() {
    let x: Box<_> = Box::new(22);
    let f = to_fn_once(move|| do_it(&*x));
    to_fn_once(move|| {
        f();
        f();
        //~^ ERROR: use of moved value: `f`
    })()
}
