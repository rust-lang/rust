#![deny(unreachable_code)]

fn main() {
    return;
    println!("Paul is dead"); //~ ERROR unreachable statement
}
