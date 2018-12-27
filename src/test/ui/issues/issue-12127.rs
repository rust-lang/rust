#![feature(box_syntax, unboxed_closures)]

fn to_fn_once<A,F:FnOnce<A>>(f: F) -> F { f }
fn do_it(x: &isize) { }

fn main() {
    let x: Box<_> = box 22;
    let f = to_fn_once(move|| do_it(&*x));
    to_fn_once(move|| {
        f();
        f();
        //~^ ERROR: use of moved value: `f`
    })()
}
