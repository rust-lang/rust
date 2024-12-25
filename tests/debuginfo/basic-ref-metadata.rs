//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:whatis ref_val
// gdb-check:type = &u8
// gdb-command:whatis mut_val
// gdb-check:type = &mut u8
// gdb-command:whatis ref_ref
// gdb-check:type = &&u8
// gdb-command:whatis mut_mut
// gdb-check:type = &mut &mut u8
// gdb-command:whatis ref_mut
// gdb-check:type = &&mut u8
// gdb-command:whatis mut_ref
// gdb-check:type = &mut &u8
// gdb-command:whatis ref_ref_ref
// gdb-check:type = &&&u8
// gdb-command:whatis ref_ref_ref_ref
// gdb-check:type = &&&&u8
// gdb-command:whatis mut_mut_mut
// gdb-check:type = &mut &mut &mut u8
// gdb-command:whatis mut_mut_mut_mut
// gdb-check:type = &mut &mut &mut &mut u8
// gdb-command:whatis ref_mut_mut
// gdb-check:type = &&mut &mut u8
// gdb-command:whatis mut_ref_ref
// gdb-check:type = &mut &&u8
// gdb-command:whatis ref_mut_ref
// gdb-check:type = &&mut &u8
// gdb-command:whatis mut_ref_mut
// gdb-check:type = &mut &&mut u8
// gdb-command:whatis ref_ptr
// gdb-check:type = &*const u8
// gdb-command:whatis ref_mut_ptr
// gdb-check:type = &*mut u8

#![allow(unused_variables)]

fn main() {
    let val1 = 1u8;
    let mut val2 = 0u8;
    let ref_val = &val1;
    let mut_val = &mut val2;
    let ref_ref = &&val1;
    let mut_mut = &mut &mut val2;
    let ref_mut = & &mut val2;
    let mut_ref = &mut &val2;

    let ref_ref_ref = &&&val1;
    let ref_ref_ref_ref = &&&&val1;
    let mut_mut_mut = &mut &mut &mut val2;
    let mut_mut_mut_mut = &mut &mut &mut &mut val2;

    let ref_mut_mut = &&mut &mut val2;
    let mut_ref_ref = &mut &&val2;
    let ref_mut_ref = & &mut &val2;
    let mut_ref_mut = &mut &&mut val2;

    let const_ptr = &val1 as *const _;
    let mut_ptr = &mut val2 as *mut _;
    let ref_ptr = &const_ptr;
    let ref_mut_ptr = &mut_ptr;

    _zzz(); // #break
}

fn _zzz() {()}
