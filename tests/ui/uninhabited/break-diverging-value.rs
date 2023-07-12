#![feature(never_type)]

fn loop_break_return() -> i32 {
    let loop_value = loop { break return 0 }; // ok
}

fn loop_break_loop() -> i32 {
    let loop_value = loop { break loop {} }; // ok
}

fn loop_break_break() -> i32 { //~ ERROR mismatched types
    let loop_value = loop { break break };
}

fn loop_break_return_2() -> i32 {
    let loop_value = loop { break { return 0; () } }; // ok
}

fn get_never() -> ! {
    panic!()
}

fn loop_break_never() -> i32 {
    let loop_value = loop { break get_never() }; // ok
}

enum Void {}

fn get_void() -> Void {
    panic!()
}

fn loop_break_void() -> i32 {
    let loop_value = loop { break get_void() }; // ok
}

struct IndirectVoid(Void);

fn get_indirect_void() -> IndirectVoid {
    panic!()
}

fn loop_break_indirect_void() -> i32 {
    let loop_value = loop { break get_indirect_void() }; // ok
}

mod private {
    pub struct PrivateVoid(super::Void);

    pub fn get_private_void() -> PrivateVoid {
        panic!()
    }
}

fn loop_break_private_void() -> i32 { //~ ERROR mismatched types
    // The field inside `PrivateVoid` is private, so the typeck is not allowed to use
    // `PrivateVoid`'s uninhabitedness to guide inference.
    let loop_value = loop { break private::get_private_void() };
}

fn main() {}
