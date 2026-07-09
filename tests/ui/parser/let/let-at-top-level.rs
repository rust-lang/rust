let x = 1;
//~^ ERROR `let` statements are not allowed outside of functions or const blocks

let y: i32 = 1;
//~^ ERROR `let` statements are not allowed outside of functions or const blocks

pub let z = 1;
//~^ ERROR `let` statements are not allowed outside of functions or const blocks

fn main() {}
