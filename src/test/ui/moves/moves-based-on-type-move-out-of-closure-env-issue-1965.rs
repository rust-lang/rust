#![feature(box_syntax, unboxed_closures)]

fn to_fn<A,F:Fn<A>>(f: F) -> F { f }

fn test(_x: Box<usize>) {}

fn main() {
    let i = box 3;
    let _f = to_fn(|| test(i)); //~ ERROR cannot move out
}
