//@ compile-flags: -Z trace-macros
//@ build-pass (FIXME(62277): could be check-pass?)

fn main() {
    println!("Hello, World!");
    //~^ NOTE trace_macro
    //~| NOTE expanding `println!
    //~| NOTE to `{
}
