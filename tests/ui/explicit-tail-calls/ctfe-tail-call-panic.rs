#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

pub const fn f() {
    become g();
}

const fn g() {
    panic!() //~ NOTE inside `g`
    //~^ NOTE in this expansion of panic!
}

const _: () = f(); //~ NOTE failed inside this call
//~^ ERROR explicit panic

fn main() {}
