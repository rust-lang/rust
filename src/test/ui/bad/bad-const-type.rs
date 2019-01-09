static i: String = 10;
//~^ ERROR mismatched types
//~| expected type `std::string::String`
//~| found type `{integer}`
//~| expected struct `std::string::String`, found integer
fn main() { println!("{}", i); }
