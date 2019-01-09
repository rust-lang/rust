enum A { B, C }

mod foo { pub fn bar() {} }

fn main() {
    match (true, false) {
        A::B => (),
        //~^ ERROR mismatched types
        //~| expected type `(bool, bool)`
        //~| found type `A`
        //~| expected tuple, found enum `A`
        _ => ()
    }
}
