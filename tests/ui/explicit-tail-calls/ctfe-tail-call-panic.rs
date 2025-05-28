#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

pub const fn f() {
    become g();
}

const fn g() {
    panic!() //~ NOTE inside `g`
    //~^ NOTE in this expansion of panic!
}

const _: () = f(); //~ NOTE evaluation of constant value failed
//~^ ERROR explicit panic

fn main() {}
