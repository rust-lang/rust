//@ check-pass
//@ compile-flags: -W unused-assignments

fn main() {
    let mut d = String::from("hello"); //~ WARN value assigned to `d` is never read
    d = String::from("ahoy");

    println!("{d}, world!");
}
