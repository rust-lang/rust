// compile-pass
// aux-build:attr_proc_macro.rs

#[macro_use] extern crate attr_proc_macro;

#[attr_proc_macro]
struct Foo;

fn main() {
    let _ = Foo;
}
