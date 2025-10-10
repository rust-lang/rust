//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// BEFORE if
// gdb-command:print x
// gdb-check:$1 = 999
// gdb-command:print y
// gdb-check:$2 = -1
// gdb-command:continue

// AT BEGINNING of 'then' block
// gdb-command:print x
// gdb-check:$3 = 999
// gdb-command:print y
// gdb-check:$4 = -1
// gdb-command:continue

// AFTER 1st redeclaration of 'x'
// gdb-command:print x
// gdb-check:$5 = 1001
// gdb-command:print y
// gdb-check:$6 = -1
// gdb-command:continue

// AFTER 2st redeclaration of 'x'
// gdb-command:print x
// gdb-check:$7 = 1002
// gdb-command:print y
// gdb-check:$8 = 1003
// gdb-command:continue

// AFTER 1st if expression
// gdb-command:print x
// gdb-check:$9 = 999
// gdb-command:print y
// gdb-check:$10 = -1
// gdb-command:continue

// BEGINNING of else branch
// gdb-command:print x
// gdb-check:$11 = 999
// gdb-command:print y
// gdb-check:$12 = -1
// gdb-command:continue

// BEGINNING of else branch
// gdb-command:print x
// gdb-check:$13 = 1004
// gdb-command:print y
// gdb-check:$14 = 1005
// gdb-command:continue

// BEGINNING of else branch
// gdb-command:print x
// gdb-check:$15 = 999
// gdb-command:print y
// gdb-check:$16 = -1
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// BEFORE if
// lldb-command:v x
// lldb-check:[...] 999
// lldb-command:v y
// lldb-check:[...] -1
// lldb-command:continue

// AT BEGINNING of 'then' block
// lldb-command:v x
// lldb-check:[...] 999
// lldb-command:v y
// lldb-check:[...] -1
// lldb-command:continue

// AFTER 1st redeclaration of 'x'
// lldb-command:v x
// lldb-check:[...] 1001
// lldb-command:v y
// lldb-check:[...] -1
// lldb-command:continue

// AFTER 2st redeclaration of 'x'
// lldb-command:v x
// lldb-check:[...] 1002
// lldb-command:v y
// lldb-check:[...] 1003
// lldb-command:continue

// AFTER 1st if expression
// lldb-command:v x
// lldb-check:[...] 999
// lldb-command:v y
// lldb-check:[...] -1
// lldb-command:continue

// BEGINNING of else branch
// lldb-command:v x
// lldb-check:[...] 999
// lldb-command:v y
// lldb-check:[...] -1
// lldb-command:continue

// BEGINNING of else branch
// lldb-command:v x
// lldb-check:[...] 1004
// lldb-command:v y
// lldb-check:[...] 1005
// lldb-command:continue

// BEGINNING of else branch
// lldb-command:v x
// lldb-check:[...] 999
// lldb-command:v y
// lldb-check:[...] -1
// lldb-command:continue

fn main() {

    let x = 999;
    let y = -1;

    zzz(); // #break
    sentinel();

    if x < 1000 {
        zzz(); // #break
        sentinel();

        let x = 1001;

        zzz(); // #break
        sentinel();

        let x = 1002;
        let y = 1003;
        zzz(); // #break
        sentinel();
    } else {
        unreachable!();
    }

    zzz(); // #break
    sentinel();

    if x > 1000 {
        unreachable!();
    } else {
        zzz(); // #break
        sentinel();

        let x = 1004;
        let y = 1005;
        zzz(); // #break
        sentinel();
    }

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
