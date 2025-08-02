//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print x
// gdb-check:$1 = false
// gdb-command:print y
// gdb-check:$2 = true
// gdb-command:continue

// gdb-command:print x
// gdb-check:$3 = 10
// gdb-command:print y
// gdb-check:$4 = true
// gdb-command:continue

// gdb-command:print x
// gdb-check:$5 = 10.5
// gdb-command:print y
// gdb-check:$6 = 20
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v x
// lldb-check:[...] false
// lldb-command:v y
// lldb-check:[...] true
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] 10
// lldb-command:v y
// lldb-check:[...] true
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] 10.5
// lldb-command:v y
// lldb-check:[...] 20
// lldb-command:continue


fn a_function(x: bool, y: bool) {
    zzz(); // #break
    sentinel();

    let x = 10;

    zzz(); // #break
    sentinel();

    let x = 10.5f64;
    let y = 20;

    zzz(); // #break
    sentinel();
}

fn main() {
    a_function(false, true);
}

fn zzz() {()}
fn sentinel() {()}
