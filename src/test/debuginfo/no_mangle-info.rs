// compile-flags:-g

// === GDB TESTS ===================================================================================
// gdb-command:run
// gdb-command:whatis TEST
// gdb-check:type = u64

// === LLDB TESTS ==================================================================================
// lldb-command:run
// lldb-command:expr TEST
// lldb-check: (unsigned long) $0 = 3735928559

// === CDB TESTS ==================================================================================
// FIXME: This does not currently work due to a bug in LLVM
// The fix for this is being tracked in rust-lang/rust#98295
// // cdb-command: g
// // cdb-command: dx a!no_mangle_info::TEST
// // cdb-check: a!no_mangle_info::TEST : 0xdeadbeef [Type: unsigned __int64]

#[no_mangle]
pub static TEST: u64 = 0xdeadbeef;

pub fn main() {
    println!("TEST: {}", TEST); // #break
}
