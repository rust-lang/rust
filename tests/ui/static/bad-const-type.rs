static i: String = 10;
//~^ ERROR mismatched types
//~| NOTE expected `String`, found integer
fn main() { println!("{}", i); }
