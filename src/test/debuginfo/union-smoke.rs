// min-lldb-version: 310
// ignore-gdb // Test temporarily ignored due to debuginfo tests being disabled, see PR 47155

// ignore-gdb-version: 7.11.90 - 7.12.9

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print u
// gdbg-check:$1 = {a = {__0 = 2 '\002', __1 = 2 '\002'}, b = 514}
// gdbr-check:$1 = union_smoke::U {a: (2, 2), b: 514}
// gdb-command:print union_smoke::SU
// gdbg-check:$2 = {a = {__0 = 1 '\001', __1 = 1 '\001'}, b = 257}
// gdbr-check:$2 = union_smoke::U {a: (1, 1), b: 257}

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print u
// lldbg-check:[...]$0 = U { a: ('\x02', '\x02'), b: 514 }
// lldbr-check:(union_smoke::U) u = { a = { = 2 = 2 } b = 514 }

// Don't test this with rust-enabled lldb for now; see
// https://github.com/rust-lang-nursery/lldb/issues/18
// lldbg-command:print union_smoke::SU
// lldbg-check:[...]$1 = U { a: ('\x01', '\x01'), b: 257 }

#![allow(unused)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

union U {
    a: (u8, u8),
    b: u16,
}

static mut SU: U = U { a: (1, 1) };

fn main() {
    let u = U { b: (2 << 8) + 2 };
    unsafe { SU = U { a: (1, 1) } }

    zzz(); // #break
}

fn zzz() {()}
