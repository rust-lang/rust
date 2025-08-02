//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print u
// gdb-check:$1 = union_smoke::U {a: (2, 2), b: 514}
// gdb-command:print union_smoke::SU
// gdb-check:$2 = union_smoke::U {a: (1, 1), b: 257}

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:v u
// lldb-check:[...] { a = { 0 = '\x02' 1 = '\x02' } b = 514 }

// lldb-command:print union_smoke::SU
// lldb-check:[...] { a = { 0 = '\x01' 1 = '\x01' } b = 257 }

#![allow(unused)]

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
