#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused_parens, unused_variables)]
#![deny(clippy, clippy_pedantic)]

fn id<T>(x: T) -> T { x }

fn first(x: (isize, isize)) -> isize { x.0 }

fn main() {
    let mut x = 1;
    let x = &mut x; //~ERROR: x is shadowed by itself in &mut x
    let x = { x }; //~ERROR: x is shadowed by itself in { x }
    let x = (&*x); //~ERROR: x is shadowed by itself in &*x
    let x = { *x + 1 }; //~ERROR: x is shadowed by { *x + 1 } which reuses
    let x = id(x); //~ERROR: x is shadowed by id(x) which reuses
    let x = (1, x); //~ERROR: x is shadowed by (1, x) which reuses
    let x = first(x); //~ERROR: x is shadowed by first(x) which reuses
    let y = 1;
    let x = y; //~ERROR: x is shadowed by y

    let o = Some(1u8);

    if let Some(p) = o { assert_eq!(1, p); }
    match o {
        Some(p) => p, // no error, because the p above is in its own scope
        None => 0,
    };

    match (x, o) {
        (1, Some(a)) | (a, Some(1)) => (), // no error though `a` appears twice
        _ => (),
    }
}
