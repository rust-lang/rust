#![feature(unboxed_closures)]

fn to_fn<A,F:Fn<A>>(f: F) -> F { f }

fn test(_x: Box<usize>) {}

fn main() {
    let i = Box::new(3);
    let _f = to_fn(|| test(i)); //~ ERROR cannot move out
}
