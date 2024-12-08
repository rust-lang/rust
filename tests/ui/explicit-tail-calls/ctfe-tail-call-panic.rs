#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

pub const fn f() {
    become g();
}

const fn g() {
    panic!()
    //~^ error: evaluation of constant value failed
    //~| note: in this expansion of panic!
    //~| note: inside `g`
    //~| note: in this expansion of panic!
}

const _: () = f();
//~^ note: inside `_`

fn main() {}
