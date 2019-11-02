const _X: i32 = if true { 5 } else { 6 };
//~^ ERROR constant contains unimplemented expression type
//~| ERROR constant contains unimplemented expression type

fn main() {}
