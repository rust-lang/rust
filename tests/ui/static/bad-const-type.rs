static i: String = 10;
//~^ ERROR mismatched types
//~| expected `String`, found integer
fn main() { println!("{}", i); }
