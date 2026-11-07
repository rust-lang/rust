//@ run-rustfix
fn main() {
    println!('1 + 1');
    //~^ ERROR unterminated character literal
    //~| ERROR lifetimes cannot start with a number
}
