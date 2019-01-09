// compile-flags:-Zborrowck=compare

#![allow(warnings)]
#![feature(rustc_attrs)]

fn main() {
}

fn nll_fail() {
    let mut data = vec!['a', 'b', 'c'];
    let slice = &mut data;
    capitalize(slice);
    data.push('d');
    //~^ ERROR (Ast) [E0499]
    //~| ERROR (Mir) [E0499]
    data.push('e');
    //~^ ERROR (Ast) [E0499]
    //~| ERROR (Mir) [E0499]
    data.push('f');
    //~^ ERROR (Ast) [E0499]
    //~| ERROR (Mir) [E0499]
    capitalize(slice);
}

fn nll_ok() {
    let mut data = vec!['a', 'b', 'c'];
    let slice = &mut data;
    capitalize(slice);
    data.push('d');
    //~^ ERROR (Ast) [E0499]
    data.push('e');
    //~^ ERROR (Ast) [E0499]
    data.push('f');
    //~^ ERROR (Ast) [E0499]
}

fn capitalize(_: &mut [char]) {
}
