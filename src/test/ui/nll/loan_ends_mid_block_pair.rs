// compile-flags:-Zborrowck=compare

#![allow(warnings)]
#![feature(rustc_attrs)]


fn main() {
}

fn nll_fail() {
    let mut data = ('a', 'b', 'c');
    let c = &mut data.0;
    capitalize(c);
    data.0 = 'e';
    //~^ ERROR (Ast) [E0506]
    //~| ERROR (Mir) [E0506]
    data.0 = 'f';
    //~^ ERROR (Ast) [E0506]
    data.0 = 'g';
    //~^ ERROR (Ast) [E0506]
    capitalize(c);
}

fn nll_ok() {
    let mut data = ('a', 'b', 'c');
    let c = &mut data.0;
    capitalize(c);
    data.0 = 'e';
    //~^ ERROR (Ast) [E0506]
    data.0 = 'f';
    //~^ ERROR (Ast) [E0506]
    data.0 = 'g';
    //~^ ERROR (Ast) [E0506]
}

fn capitalize(_: &mut char) {
}
