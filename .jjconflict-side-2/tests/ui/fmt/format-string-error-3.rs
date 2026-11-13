//@ run-rustfix
fn main() {
    let x = 32;
    println!("{=}", x); //~ ERROR invalid format string: python's f-string debug
}
