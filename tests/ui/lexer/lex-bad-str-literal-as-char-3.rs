//@ revisions: rust2015 rust2018 rust2021
//@[rust2015] edition:2015
//@[rust2018] edition:2018
//@[rust2021] edition:2021
fn main() {
    println!('hello world');
    //~^ ERROR unterminated character literal
    //[rust2021]~| ERROR prefix `world` is unknown
}
