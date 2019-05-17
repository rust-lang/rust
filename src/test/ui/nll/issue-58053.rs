#![feature(nll)]

fn main() {
    let i = &3;

    let f = |x: &i32| -> &i32 { x };
    //~^ ERROR lifetime may not live long enough
    let j = f(i);

    let g = |x: &i32| { x };
    //~^ ERROR lifetime may not live long enough
    let k = g(i);
}
