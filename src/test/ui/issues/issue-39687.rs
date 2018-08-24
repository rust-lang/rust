#![feature(fn_traits)]

fn main() {
    <fn() as Fn()>::call;
    //~^ ERROR associated type bindings are not allowed here [E0229]
}
