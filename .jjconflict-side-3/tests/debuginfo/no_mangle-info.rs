//@ revisions: msvc non-msvc

//@ [msvc] only-msvc
//@ [non-msvc] ignore-msvc

//@ compile-flags:-g --crate-name=no_mangle_info
//@ min-gdb-version: 10.1
//@ ignore-backends: gcc

// === GDB TESTS ===================================================================================
//@ gdb-command:run
//@ gdb-command:p TEST
//@ gdb-check:$1 = 3735928559
//@ gdb-command:p no_mangle_info::namespace::OTHER_TEST
//@ gdb-check:$2 = 42

// === LLDB TESTS ==================================================================================
//@ lldb-command:run
//@ [msvc] lldb-command:v no_mangle_info::TEST
//@ [msvc] lldb-check:[...]no_mangle_info::TEST = 3735928559

//@ [non-msvc] lldb-command:v TEST
//@ [non-msvc] lldb-check:[...]TEST = 3735928559
//@ lldb-command:v no_mangle_info::namespace::OTHER_TEST
//@ lldb-check:[...]no_mangle_info::namespace::OTHER_TEST = 42

// === CDB TESTS ==================================================================================
//@ cdb-command: g
// Note: LLDB and GDB allow referring to items that are in the same namespace of the symbol
// we currently have a breakpoint on in an unqualified way. CDB does not, and thus we need to
// refer to it in a fully qualified way.
//@ cdb-command: dx a!no_mangle_info::TEST
//@ cdb-check: a!no_mangle_info::TEST : 0xdeadbeef [Type: unsigned __int64]
//@ cdb-command: dx a!no_mangle_info::namespace::OTHER_TEST
//@ cdb-check: a!no_mangle_info::namespace::OTHER_TEST : 0x2a [Type: unsigned __int64]

#[no_mangle]
pub static TEST: u64 = 0xdeadbeef;

// FIXME(rylev, wesleywiser): uncommenting this item breaks the test, and we're not sure why
// pub static OTHER_TEST: u64 = 43;
pub mod namespace {
    pub static OTHER_TEST: u64 = 42;
}

pub fn main() {
    println!("TEST: {}", unsafe { TEST} );
    println!("OTHER TEST: {}", namespace::OTHER_TEST); // #break
}
