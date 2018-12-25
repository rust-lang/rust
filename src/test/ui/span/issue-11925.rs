#![feature(box_syntax, unboxed_closures)]

fn to_fn_once<A,F:FnOnce<A>>(f: F) -> F { f }

fn main() {
    let r = {
        let x: Box<_> = box 42;
        let f = to_fn_once(move|| &x); //~ ERROR does not live long enough
        f()
    };

    drop(r);
}
