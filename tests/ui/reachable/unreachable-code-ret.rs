//@error-in-other-file: unreachable statement

#![deny(unreachable_code)]

fn main() {
    return;
    println!("Paul is dead");
}
