//@ run-pass

//@ compile-flags: -Clto=thin
//@ no-prefer-dynamic
//@ ignore-backends: gcc

fn main() {
    println!("hello!");
}
