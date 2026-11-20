// issue rust-lang/rust#121463
// ICE non-ADT in struct pattern

fn main() {
    let mut a = E::StructVar { boxed: Box::new(5_i32) };
    //~^ ERROR cannot find type `E`
    match a {
        E::StructVar { boxed } => { }
        //~^ ERROR cannot find type `E`
    }
}
