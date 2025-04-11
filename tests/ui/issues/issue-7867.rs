enum A { B, C }

mod foo { pub fn bar() {} }

fn main() {
    match (true, false) {
        A::B => (),
        //~^ ERROR mismatched types
        //~| NOTE_NONVIRAL expected `(bool, bool)`, found `A`
        //~| NOTE_NONVIRAL expected tuple `(bool, bool)`
        //~| NOTE_NONVIRAL found enum `A`
        _ => ()
    }
}
