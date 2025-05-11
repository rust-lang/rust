//@ ignore-android

// FIXME: stepping with "next" in a debugger skips past end-of-scope drops
//@ ignore-test (broken, see #128971)

#![allow(unused)]

//@ compile-flags:-g

// This test checks that drop glue code gets attributed to scope's closing brace,
// and function epilogues - to function's closing brace.

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc1[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc2[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc3[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc4[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc5[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc6[...]

// === LLDB TESTS ==================================================================================

// lldb-command:set set stop-line-count-before 0
// lldb-command:set set stop-line-count-after 1
// Can't set both to zero or lldb will stop printing source at all.  So it will output the current
// line and the next.  We deal with this by having at least 2 lines between the #loc's

// lldb-command:run
// lldb-command:next
// lldb-command:frame select
// lldb-check:[...] #loc1 [...]
// lldb-command:next
// lldb-command:frame select
// lldb-check:[...] #loc2 [...]
// lldb-command:next
// lldb-command:frame select
// lldb-check:[...] #loc3 [...]
// lldb-command:next
// lldb-command:frame select
// lldb-check:[...] #loc4 [...]
// lldb-command:next
// lldb-command:frame select
// lldb-check:[...] #loc5 [...]
// lldb-command:next
// lldb-command:frame select
// lldb-check:[...] #loc6 [...]

fn main() {

    foo();

    zzz(); // #loc5

} // #loc6

fn foo() {
    {
        let s = String::from("s"); // #break

        zzz(); // #loc1

    } // #loc2

    zzz(); // #loc3

} // #loc4

fn zzz() {()}
