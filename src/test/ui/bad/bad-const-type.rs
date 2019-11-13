static i: String = 10;
//~^ ERROR mismatched types
//~| expected struct `std::string::String`, found integer
//~| expected struct `std::string::String`
//~| found type `{integer}`
fn main() { println!("{}", i); }
