//@ run-pass
fn main() {
    println!("{}", 0E+10);
    println!("{}", 0e+10);
    println!("{}", 00e+10);
    println!("{}", 00E+10);
}
