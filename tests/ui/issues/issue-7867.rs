enum A { B, C }

mod foo { pub fn bar() {} }

fn main() {
    match (true, false) {
        A::B => (),
        //~^ ERROR mismatched types
        //~| expected `(bool, bool)`, found `A`
        //~| expected tuple `(bool, bool)`
        //~| found enum `A`
        _ => ()
    }
}
