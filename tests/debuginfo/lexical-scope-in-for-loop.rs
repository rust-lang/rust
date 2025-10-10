//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// FIRST ITERATION
// gdb-command:print x
// gdb-check:$1 = 1
// gdb-command:continue

// gdb-command:print x
// gdb-check:$2 = -1
// gdb-command:continue

// SECOND ITERATION
// gdb-command:print x
// gdb-check:$3 = 2
// gdb-command:continue

// gdb-command:print x
// gdb-check:$4 = -2
// gdb-command:continue

// THIRD ITERATION
// gdb-command:print x
// gdb-check:$5 = 3
// gdb-command:continue

// gdb-command:print x
// gdb-check:$6 = -3
// gdb-command:continue

// AFTER LOOP
// gdb-command:print x
// gdb-check:$7 = 1000000
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// FIRST ITERATION
// lldb-command:v x
// lldb-check:[...] 1
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] -1
// lldb-command:continue

// SECOND ITERATION
// lldb-command:v x
// lldb-check:[...] 2
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] -2
// lldb-command:continue

// THIRD ITERATION
// lldb-command:v x
// lldb-check:[...] 3
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] -3
// lldb-command:continue

// AFTER LOOP
// lldb-command:v x
// lldb-check:[...] 1000000
// lldb-command:continue

fn main() {

    let range = [1, 2, 3];

    let x = 1000000; // wan meeeljen doollaars!

    for &x in &range {
        zzz(); // #break
        sentinel();

        let x = -1 * x;

        zzz(); // #break
        sentinel();
    }

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
