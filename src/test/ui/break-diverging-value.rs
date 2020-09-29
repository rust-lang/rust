fn loop_break_return() -> i32 {
    let loop_value = loop { break return 0 }; // ok
}

fn loop_break_loop() -> i32 {
    let loop_value = loop { break loop {} }; // ok
}

fn loop_break_break() -> i32 { //~ ERROR mismatched types
    let loop_value = loop { break break };
}

fn main() {}
