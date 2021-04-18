static i: String = 10;
//~^ ERROR mismatched types
//~| expected struct `String`, found integer
fn main() { println!("{}", i); }
