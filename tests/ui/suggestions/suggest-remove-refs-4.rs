//@ run-rustfix
fn main() {
    let foo = &[1,2,3].iter();
    for _i in &foo {} //~ ERROR E0277
}
