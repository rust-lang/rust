// Test ensuring that `dbg!(expr)` will take ownership of the argument.

#![feature(dbg_macro)]

#[derive(Debug)]
struct NoCopy(usize);

fn main() {
    let a = NoCopy(0);
    let _ = dbg!(a);
    let _ = dbg!(a);
}
