const A: usize = { 1; 2 };
//~^ ERROR statements in constants are unstable

const B: usize = { { } 2 };
//~^ ERROR statements in constants are unstable

macro_rules! foo {
    () => (()) //~ ERROR statements in constants are unstable
}
const C: usize = { foo!(); 2 };

const D: usize = { let x = 4; 2 };
//~^ ERROR let bindings in constants are unstable
//~| ERROR statements in constants are unstable
//~| ERROR let bindings in constants are unstable
//~| ERROR statements in constants are unstable

pub fn main() {}
