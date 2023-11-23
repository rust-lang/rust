// aux-build:format_in_macro.rs
extern crate format_in_macro;
#[derive(format_in_macro::derive)]
pub struct Foo;
//~^^ERROR format argument must be a string literal

format_in_macro::function!();
//~^ERROR format argument must be a string literal

#[format_in_macro::attribute]
const UNIT:() = ();
//~^^ERROR format argument must be a string literal

macro_rules! indirect {
    () => {
        format_in_macro::function!();
        //~^ERROR format argument must be a string literal
    }
}
fn new_scope() {
    indirect!();
}

fn main() {}
