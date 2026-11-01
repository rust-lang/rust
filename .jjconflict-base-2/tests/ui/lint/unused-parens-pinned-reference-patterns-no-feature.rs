#![feature(mut_ref)]
#![warn(unused_parens)]

fn main() {
    let pin_const: &pin const i32 = todo!();
    //~^ ERROR pinned reference syntax is experimental
    let &pin const (_x) = pin_const;
    //~^ ERROR pinned reference syntax is experimental
}
