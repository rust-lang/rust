// Test ensuring that `dbg!(expr)` requires the passed type to implement `Debug`.

#![feature(dbg_macro)]

struct NotDebug;

fn main() {
    let _: NotDebug = dbg!(NotDebug);
}
