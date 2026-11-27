//@ check-pass

macro_rules! macro_rules { () => { struct S; } } // OK

macro_rules! {} // OK, calls the macro defined above

fn main() {
    let s = S;
}
