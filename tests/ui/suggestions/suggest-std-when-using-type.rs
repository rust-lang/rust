//@ run-rustfix
fn main() {
    let pi = f32::consts::PI; //~ ERROR ambiguous associated type
    println!("{pi}");
}
