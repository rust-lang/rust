// issue rust-lang/rust#121463
// ICE non-ADT in struct pattern
#![feature(box_patterns)]

fn main() {
    let mut a = E::StructVar { boxed: Box::new(5_i32) };
    //~^ ERROR failed to resolve: use of undeclared type `E`
    match a {
        E::StructVar { box boxed } => { }
        //~^ ERROR failed to resolve: use of undeclared type `E`
    }
}
