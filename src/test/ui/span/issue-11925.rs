#![feature(unboxed_closures)]

fn to_fn_once<A,F:FnOnce<A>>(f: F) -> F { f }

fn main() {
    let r = {
        let x: Box<_> = Box::new(42);
        let f = to_fn_once(move|| &x); //~ ERROR cannot return reference to local data `x`
        f()
    };

    drop(r);
}
