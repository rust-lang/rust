//@ ignore-windows-gnu: #128981
//@ ignore-android: FIXME(#10381)
//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print empty_string
// gdb-check:$1 = ""

// gdb-command: print empty_str
// gdb-check:$2 = ""

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:fr v empty_string
// lldb-check:[...] empty_string = ""

// lldb-command:fr v empty_str
// lldb-check:[...] empty_str = ""

fn main() {
    let empty_string = String::new();

    let empty_str = "";

    zzz(); // #break
}

fn zzz() {}
