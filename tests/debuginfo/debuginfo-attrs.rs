//@ compile-flags:-g

// gate-test-debuginfo_attrs
// Tests the `#[debuginfo_transparent]` attribute.

// === CDB TESTS ==================================================================================
// cdb-command: g

// cdb-command: dx transparent
// cdb-check:transparent           : 1 [Type: u32]
// cdb-check:    [<Raw View>]     [Type: u32]

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print transparent
// gdb-check:[...]$1 = 1

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v transparent
// lldb-check:[...] 1

#![feature(debuginfo_attrs)]

#[repr(transparent)]
#[debuginfo_transparent]
struct Transparent(u32);

fn main() {
    let transparent = Transparent(1);

    zzz(); // #break
}

fn zzz() {}
