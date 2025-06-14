//! Tests type mismatches with `break` and diverging types in loops

#![feature(never_type)]

fn loop_break_return() -> i32 {
    let loop_value = loop {
        break return 0;
    }; // ok
}

fn loop_break_loop() -> i32 {
    let loop_value = loop {
        break loop {};
    }; // ok
}

fn loop_break_break() -> i32 {
    //~^ ERROR mismatched types
    let loop_value = loop {
        break break;
    };
}

fn loop_break_return_2() -> i32 {
    let loop_value = loop {
        break {
            return 0;
            ()
        };
    }; // ok
}

enum Void {}

fn get_void() -> Void {
    panic!()
}

fn loop_break_void() -> i32 {
    //~^ ERROR mismatched types
    let loop_value = loop {
        break get_void();
    };
}

fn get_never() -> ! {
    panic!()
}

fn loop_break_never() -> i32 {
    let loop_value = loop {
        break get_never();
    }; // ok
}

fn main() {}
