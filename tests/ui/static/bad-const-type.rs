static i: String = 10;
//~^ ERROR mismatched types
//~| NOTE expected `String`, found integer
//~| NOTE expected struct `String`
fn main() { println!("{}", i); }
