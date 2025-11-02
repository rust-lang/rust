// https://github.com/rust-lang/rust/issues/7867
//@ dont-require-annotations: NOTE

enum A { B, C }

mod foo { pub fn bar() {} }

fn main() {
    match (true, false) {
        A::B => (),
        //~^ ERROR mismatched types
        //~| NOTE expected `(bool, bool)`, found `A`
        //~| NOTE expected tuple `(bool, bool)`
        //~| NOTE found enum `A`
        _ => ()
    }
}
