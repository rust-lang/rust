#![feature(fn_traits)]

fn main() {
    <fn() as Fn()>::call;
    //~^ ERROR associated item constraints are not allowed here [E0229]
}
