//@ run-pass
//@ compile-flags: -Clto=fat
//@ no-prefer-dynamic
//@ ignore-backends: gcc

fn main() {
    println!("hello!");
}
