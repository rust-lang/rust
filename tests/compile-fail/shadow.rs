#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused_parens, unused_variables)]
#![deny(shadow)]

fn id<T>(x: T) -> T { x }

fn first(x: (isize, isize)) -> isize { x.0 }

fn main() {
    let mut x = 1;
    let x = &mut x; //~ERROR: x is shadowed by itself in &mut x
    let x = { x }; //~ERROR: x is shadowed by itself in { x }
    let x = (&*x); //~ERROR: x is shadowed by itself in (&*x)
    let x = { *x + 1 }; //~ERROR: x is shadowed by { *x + 1 } which reuses
    let x = id(x); //~ERROR: x is shadowed by id(x) which reuses
    let x = (1, x); //~ERROR: x is shadowed by (1, x) which reuses
    let x = first(x); //~ERROR: x is shadowed by first(x) which reuses
    let y = 1;
    let x = y; //~ERROR: x is shadowed by y in this declaration
}
