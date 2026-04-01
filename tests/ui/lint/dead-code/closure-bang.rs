#![deny(unreachable_code)]

fn main() {
    let x = || -> ! { panic!() };
    x();
    println!("Foo bar"); //~ ERROR: unreachable statement
}
