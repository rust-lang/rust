// error-pattern: unreachable statement

#![deny(unreachable_code)]

fn main() {
    return;
    println!("Paul is dead");
}
