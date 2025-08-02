//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print a
// gdb-check:$1 = 10
// gdb-command:print b
// gdb-check:$2 = 34
// gdb-command:continue

// gdb-command:print a
// gdb-check:$3 = 890242
// gdb-command:print b
// gdb-check:$4 = 34
// gdb-command:continue

// gdb-command:print a
// gdb-check:$5 = 10
// gdb-command:print b
// gdb-check:$6 = 34
// gdb-command:continue

// gdb-command:print a
// gdb-check:$7 = 102
// gdb-command:print b
// gdb-check:$8 = 34
// gdb-command:continue

// gdb-command:print a
// gdb-check:$9 = 110
// gdb-command:print b
// gdb-check:$10 = 34
// gdb-command:continue

// gdb-command:print a
// gdb-check:$11 = 10
// gdb-command:print b
// gdb-check:$12 = 34
// gdb-command:continue

// gdb-command:print a
// gdb-check:$13 = 10
// gdb-command:print b
// gdb-check:$14 = 34
// gdb-command:print c
// gdb-check:$15 = 400
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v a
// lldb-check:[...] 10
// lldb-command:v b
// lldb-check:[...] 34
// lldb-command:continue

// lldb-command:v a
// lldb-check:[...] 890242
// lldb-command:v b
// lldb-check:[...] 34
// lldb-command:continue

// lldb-command:v a
// lldb-check:[...] 10
// lldb-command:v b
// lldb-check:[...] 34
// lldb-command:continue

// lldb-command:v a
// lldb-check:[...] 102
// lldb-command:v b
// lldb-check:[...] 34
// lldb-command:continue

// lldb-command:print a
// lldb-check:[...] 110
// lldb-command:print b
// lldb-check:[...] 34
// lldb-command:continue

// lldb-command:print a
// lldb-check:[...] 10
// lldb-command:print b
// lldb-check:[...] 34
// lldb-command:continue

// lldb-command:print a
// lldb-check:[...] 10
// lldb-command:print b
// lldb-check:[...] 34
// lldb-command:print c
// lldb-check:[...] 400
// lldb-command:continue

macro_rules! trivial {
    ($e1:expr) => ($e1)
}

macro_rules! no_new_scope {
    ($e1:expr) => (($e1 + 2) - 1)
}

macro_rules! new_scope {
    () => ({
        let a = 890242;
        zzz(); // #break
        sentinel();
    })
}

macro_rules! shadow_within_macro {
    ($e1:expr) => ({
        let a = $e1 + 2;

        zzz(); // #break
        sentinel();

        let a = $e1 + 10;

        zzz(); // #break
        sentinel();
    })
}


macro_rules! dup_expr {
    ($e1:expr) => (($e1) + ($e1))
}


fn main() {

    let a = trivial!(10);
    let b = no_new_scope!(33);

    zzz(); // #break
    sentinel();

    new_scope!();

    zzz(); // #break
    sentinel();

    shadow_within_macro!(100);

    zzz(); // #break
    sentinel();

    let c = dup_expr!(10 * 20);

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
