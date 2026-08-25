//@edition:2021
macro_rules! foo {
    () => {
        println!('hello world');
        //~^ ERROR unterminated character literal
        //~| ERROR prefix `world` is unknown
    }
}
fn main() {}
