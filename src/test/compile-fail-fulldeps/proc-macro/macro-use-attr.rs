// aux-build:attr_proc_macro.rs

#[macro_use] extern crate attr_proc_macro;

#[attr_proc_macro]
//~^ ERROR: attribute procedural macros cannot be imported with `#[macro_use]`
struct Foo;

fn main() {
    let _ = Foo;
}
