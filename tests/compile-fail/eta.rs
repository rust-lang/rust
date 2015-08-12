#![feature(plugin)]
#![plugin(clippy)]
#![allow(unknown_lints, unused)]
#![deny(redundant_closure)]

fn main() {
    let a = |a, b| foo(a, b);
    //~^ ERROR redundant closure found. Consider using `foo` in its place
    let c = |a, b| {1+2; foo}(a, b);
    //~^ ERROR redundant closure found. Consider using `{ 1 + 2; foo }` in its place
    let d = |a, b| foo((|c, d| foo2(c,d))(a,b), b);
    //~^ ERROR redundant closure found. Consider using `foo2` in its place
}

fn foo(_: u8, _: u8) {

}

fn foo2(_: u8, _: u8) -> u8 {
    1u8
}
