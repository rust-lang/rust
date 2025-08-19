//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print shadowed
// gdb-check:$1 = 231
// gdb-command:print not_shadowed
// gdb-check:$2 = 232
// gdb-command:continue

// gdb-command:print shadowed
// gdb-check:$3 = 233
// gdb-command:print not_shadowed
// gdb-check:$4 = 232
// gdb-command:print local_to_arm
// gdb-check:$5 = 234
// gdb-command:continue

// gdb-command:print shadowed
// gdb-check:$6 = 236
// gdb-command:print not_shadowed
// gdb-check:$7 = 232
// gdb-command:continue

// gdb-command:print shadowed
// gdb-check:$8 = 237
// gdb-command:print not_shadowed
// gdb-check:$9 = 232
// gdb-command:print local_to_arm
// gdb-check:$10 = 238
// gdb-command:continue

// gdb-command:print shadowed
// gdb-check:$11 = 239
// gdb-command:print not_shadowed
// gdb-check:$12 = 232
// gdb-command:continue

// gdb-command:print shadowed
// gdb-check:$13 = 241
// gdb-command:print not_shadowed
// gdb-check:$14 = 232
// gdb-command:continue

// gdb-command:print shadowed
// gdb-check:$15 = 243
// gdb-command:print *local_to_arm
// gdb-check:$16 = 244
// gdb-command:continue

// gdb-command:print shadowed
// gdb-check:$17 = 231
// gdb-command:print not_shadowed
// gdb-check:$18 = 232
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v shadowed
// lldb-check:[...] 231
// lldb-command:v not_shadowed
// lldb-check:[...] 232
// lldb-command:continue

// lldb-command:v shadowed
// lldb-check:[...] 233
// lldb-command:v not_shadowed
// lldb-check:[...] 232
// lldb-command:v local_to_arm
// lldb-check:[...] 234
// lldb-command:continue

// lldb-command:v shadowed
// lldb-check:[...] 236
// lldb-command:v not_shadowed
// lldb-check:[...] 232
// lldb-command:continue

// lldb-command:v shadowed
// lldb-check:[...] 237
// lldb-command:v not_shadowed
// lldb-check:[...] 232
// lldb-command:v local_to_arm
// lldb-check:[...] 238
// lldb-command:continue

// lldb-command:v shadowed
// lldb-check:[...] 239
// lldb-command:v not_shadowed
// lldb-check:[...] 232
// lldb-command:continue

// lldb-command:v shadowed
// lldb-check:[...] 241
// lldb-command:v not_shadowed
// lldb-check:[...] 232
// lldb-command:continue

// lldb-command:v shadowed
// lldb-check:[...] 243
// lldb-command:v *local_to_arm
// lldb-check:[...] 244
// lldb-command:continue

// lldb-command:v shadowed
// lldb-check:[...] 231
// lldb-command:v not_shadowed
// lldb-check:[...] 232
// lldb-command:continue

struct Struct {
    x: isize,
    y: isize
}

fn main() {

    let shadowed = 231;
    let not_shadowed = 232;

    zzz(); // #break
    sentinel();

    match (233, 234) {
        (shadowed, local_to_arm) => {

            zzz(); // #break
            sentinel();
        }
    }

    match (235, 236) {
        // with literal
        (235, shadowed) => {

            zzz(); // #break
            sentinel();
        }
        _ => {}
    }

    match (Struct { x: 237, y: 238 }) {
        Struct { x: shadowed, y: local_to_arm } => {

            zzz(); // #break
            sentinel();
        }
    }

    match (Struct { x: 239, y: 240 }) {
        // ignored field
        Struct { x: shadowed, .. } => {

            zzz(); // #break
            sentinel();
        }
    }

    match (Struct { x: 241, y: 242 }) {
        // with literal
        Struct { x: shadowed, y: 242 } => {

            zzz(); // #break
            sentinel();
        }
        _ => {}
    }

    match (243, 244) {
        (shadowed, ref local_to_arm) => {

            zzz(); // #break
            sentinel();
        }
    }

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
