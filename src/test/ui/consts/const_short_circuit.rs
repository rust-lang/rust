#![feature(underscore_const_names, const_let)]

const _: bool = false && false; //~ WARN boolean short circuiting operators in constants
const _: bool = true && false; //~ WARN boolean short circuiting operators in constants
const _: bool = {
    let mut x = true && false; //~ WARN boolean short circuiting operators in constants
    //~^ ERROR short circuiting operators do not actually short circuit in constant
    x
};
const _: bool = {
    let x = true && false; //~ WARN boolean short circuiting operators in constants
    //~^ ERROR short circuiting operators do not actually short circuit in constant
    x
};

fn main() {}
