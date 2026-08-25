// Test ensuring that `dbg!(expr)` requires the passed type to implement `Debug`.
//
// `dbg!` shouldn't tell the user about format literal syntax; the user didn't write one.
//@ forbid-output: cannot be formatted using

struct NotDebug;

fn main() {
    let _: NotDebug = dbg!(NotDebug); //~ ERROR `NotDebug` doesn't implement `Debug`
}
