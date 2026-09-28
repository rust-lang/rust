//@ compile-flags: -Z trace-macros
//@ check-pass

fn main() {
    println!("Hello, World!");
    //~^ NOTE trace_macro
    //~| NOTE expanding `println!
    //~| NOTE to `{
}
