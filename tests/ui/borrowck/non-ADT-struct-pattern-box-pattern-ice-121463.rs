// issue rust-lang/rust#121463
// ICE non-ADT in struct pattern
#![feature(box_patterns)]

fn main() {
    let mut a = E::StructVar { boxed: Box::new(5_i32) };
    //~^ ERROR cannot find item `E` in this scope
    match a {
        E::StructVar { box boxed } => { }
        //~^ ERROR cannot find item `E` in this scope
    }
}
