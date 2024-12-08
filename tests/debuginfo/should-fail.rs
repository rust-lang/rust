// == Test [gdb|lldb]-[command|check] are parsed correctly ===
//@ should-fail
//@ needs-run-enabled
//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print x
// gdb-check:$1 = 5

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v x
// lldb-check:[...] 5

// === CDB TESTS ==================================================================================

// cdb-command:g

// cdb-command:dx x
// cdb-check:string [...] : 5 [Type: [...]]

fn main() {
    let x = 1;

    zzz(); // #break
}

fn zzz() {()}
