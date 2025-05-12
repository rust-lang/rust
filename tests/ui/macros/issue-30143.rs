use std::fmt::Write;

fn main() {
    println!(0);
    //~^ ERROR format argument must be a string literal
    eprintln!('a');
    //~^ ERROR format argument must be a string literal
    let mut s = String::new();
    writeln!(s, true).unwrap();
    //~^ ERROR format argument must be a string literal
}
