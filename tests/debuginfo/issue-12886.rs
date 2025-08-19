//@ ignore-lldb
//@ ignore-aarch64

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// gdb-command:run
// gdb-command:next
// gdb-check:[...]22[...]let s = Some(5).unwrap(); // #break
// gdb-command:continue


// IF YOU MODIFY THIS FILE, BE CAREFUL TO ADAPT THE LINE NUMBERS IN THE DEBUGGER COMMANDS

// This test makes sure that gdb does not set unwanted breakpoints in inlined functions. If a
// breakpoint existed in unwrap(), then calling `next` would (when stopped at `let s = ...`) stop
// in unwrap() instead of stepping over the function invocation. By making sure that `s` is
// contained in the output, after calling `next` just once, we can be sure that we did not stop in
// unwrap(). (The testing framework doesn't allow for checking that some text is *not* contained in
// the output, which is why we have to make the test in this kind of roundabout way)
fn bar() -> isize {
    let s = Some(5).unwrap(); // #break
    s
}

fn main() {
    let _ = bar();
}
