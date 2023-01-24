#![deny(unreachable_patterns)]

fn main() {
    let s: &[bool] = &[];

    match s {
        [true, ..] => {}
        [true, ..] => {}
        //~^ ERROR multiple unreachable patterns
        //~| this arm is never executed
        [true] => {}
        //~^ this arm is never executed
        [..] => {}
    }
    match s {
        [.., true] => {}
        [.., true] => {}
        //~^ ERROR multiple unreachable patterns
        //~| this arm is never executed
        [true] => {}
        //~^ this arm is never executed
        [..] => {}
    }
    match s {
        [false, .., true] => {}
        [false, .., true] => {}
        //~^ ERROR multiple unreachable patterns
        //~| this arm is never executed
        [false, true] => {}
        //~^ this arm is never executed
        [false] => {}
        [..] => {}
    }
}
