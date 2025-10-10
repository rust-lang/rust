//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print x
// gdb-check:$1 = false
// gdb-command:continue

// gdb-command:print x
// gdb-check:$2 = false
// gdb-command:continue

// gdb-command:print x
// gdb-check:$3 = 1000
// gdb-command:continue

// gdb-command:print x
// gdb-check:$4 = 2.5
// gdb-command:continue

// gdb-command:print x
// gdb-check:$5 = true
// gdb-command:continue

// gdb-command:print x
// gdb-check:$6 = false
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v x
// lldb-check:[...] false
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] false
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] 1000
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] 2.5
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] true
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] false
// lldb-command:continue


fn main() {

    let x = false;

    zzz(); // #break
    sentinel();

    let unique_closure = |x:isize| {
        zzz(); // #break
        sentinel();

        let x = 2.5f64;

        zzz(); // #break
        sentinel();

        let x = true;

        zzz(); // #break
        sentinel();
    };

    zzz(); // #break
    sentinel();

    unique_closure(1000);

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
