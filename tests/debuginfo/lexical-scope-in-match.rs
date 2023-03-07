// min-lldb-version: 310

// compile-flags:-g

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

// lldb-command:print shadowed
// lldbg-check:[...]$0 = 231
// lldbr-check:(i32) shadowed = 231
// lldb-command:print not_shadowed
// lldbg-check:[...]$1 = 232
// lldbr-check:(i32) not_shadowed = 232
// lldb-command:continue

// lldb-command:print shadowed
// lldbg-check:[...]$2 = 233
// lldbr-check:(i32) shadowed = 233
// lldb-command:print not_shadowed
// lldbg-check:[...]$3 = 232
// lldbr-check:(i32) not_shadowed = 232
// lldb-command:print local_to_arm
// lldbg-check:[...]$4 = 234
// lldbr-check:(i32) local_to_arm = 234
// lldb-command:continue

// lldb-command:print shadowed
// lldbg-check:[...]$5 = 236
// lldbr-check:(i32) shadowed = 236
// lldb-command:print not_shadowed
// lldbg-check:[...]$6 = 232
// lldbr-check:(i32) not_shadowed = 232
// lldb-command:continue

// lldb-command:print shadowed
// lldbg-check:[...]$7 = 237
// lldbr-check:(isize) shadowed = 237
// lldb-command:print not_shadowed
// lldbg-check:[...]$8 = 232
// lldbr-check:(i32) not_shadowed = 232
// lldb-command:print local_to_arm
// lldbg-check:[...]$9 = 238
// lldbr-check:(isize) local_to_arm = 238
// lldb-command:continue

// lldb-command:print shadowed
// lldbg-check:[...]$10 = 239
// lldbr-check:(isize) shadowed = 239
// lldb-command:print not_shadowed
// lldbg-check:[...]$11 = 232
// lldbr-check:(i32) not_shadowed = 232
// lldb-command:continue

// lldb-command:print shadowed
// lldbg-check:[...]$12 = 241
// lldbr-check:(isize) shadowed = 241
// lldb-command:print not_shadowed
// lldbg-check:[...]$13 = 232
// lldbr-check:(i32) not_shadowed = 232
// lldb-command:continue

// lldb-command:print shadowed
// lldbg-check:[...]$14 = 243
// lldbr-check:(i32) shadowed = 243
// lldb-command:print *local_to_arm
// lldbg-check:[...]$15 = 244
// lldbr-check:(i32) *local_to_arm = 244
// lldb-command:continue

// lldb-command:print shadowed
// lldbg-check:[...]$16 = 231
// lldbr-check:(i32) shadowed = 231
// lldb-command:print not_shadowed
// lldbg-check:[...]$17 = 232
// lldbr-check:(i32) not_shadowed = 232
// lldb-command:continue

#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

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
