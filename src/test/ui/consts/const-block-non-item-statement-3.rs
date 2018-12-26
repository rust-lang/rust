type Array = [u32; {  let x = 2; 5 }];
//~^ ERROR let bindings in constants are unstable
//~| ERROR statements in constants are unstable
//~| ERROR let bindings in constants are unstable
//~| ERROR statements in constants are unstable

pub fn main() {}
