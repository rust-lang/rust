#![deny(unreachable_patterns)]

fn main() {
    let s: &[bool] = &[];

    match s {
        [true, ..] => {}
        [true, ..] => {}
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
        [true] => {}
        //~^ this arm is never executed
        [..] => {}
    }
    match s {
        [.., true] => {}
        [.., true] => {}
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
        [true] => {}
        //~^ this arm is never executed
        [..] => {}
    }
    match s {
        [false, .., true] => {}
        [false, .., true] => {}
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
        [false, true] => {}
        //~^ this arm is never executed
        [false] => {}
        [..] => {}
    }
}
