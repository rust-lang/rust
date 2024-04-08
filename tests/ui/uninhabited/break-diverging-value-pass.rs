//@ check-pass
#![feature(never_type)]

fn loop_break_return() -> i32 {
    let loop_value = loop { break return 0 }; // ok
}

fn loop_break_loop() -> i32 {
    let loop_value = loop { break loop {} }; // ok
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

fn main() {}
