struct S<'1> { s: &'1 usize }
//~^ ERROR lifetimes cannot start with a number
//~| ERROR lifetimes cannot start with a number
fn main() {
    // verify that the parse error doesn't stop type checking
    let x: usize = "";
    //~^ ERROR mismatched types
}
