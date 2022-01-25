// ignore-windows
// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print empty.length
// gdb-check:$1 = 0

// gdb-command:print singleton.length
// gdb-check:$2 = 1
// gdbg-command:print *((i64[1]*)(singleton.data_ptr))
// gdbr-command:print *(singleton.data_ptr as *const [i64; 1])
// gdbg-check:$3 = {1}
// gdbr-check:$3 = [1]

// gdb-command:print multiple.length
// gdb-check:$4 = 4
// gdbg-command:print *((i64[4]*)(multiple.data_ptr))
// gdbr-command:print *(multiple.data_ptr as *const [i64; 4])
// gdbg-check:$5 = {2, 3, 4, 5}
// gdbr-check:$5 = [2, 3, 4, 5]

// gdb-command:print slice_of_slice.length
// gdb-check:$6 = 2
// gdbg-command:print *((i64[2]*)(slice_of_slice.data_ptr))
// gdbr-command:print *(slice_of_slice.data_ptr as *const [i64; 2])
// gdbg-check:$7 = {3, 4}
// gdbr-check:$7 = [3, 4]

// gdb-command:print padded_tuple.length
// gdb-check:$8 = 2
// gdb-command:print padded_tuple.data_ptr[0]
// gdbg-check:$9 = {__0 = 6, __1 = 7}
// gdbr-check:$9 = (6, 7)
// gdb-command:print padded_tuple.data_ptr[1]
// gdbg-check:$10 = {__0 = 8, __1 = 9}
// gdbr-check:$10 = (8, 9)

// gdb-command:print padded_struct.length
// gdb-check:$11 = 2
// gdb-command:print padded_struct.data_ptr[0]
// gdbg-check:$12 = {x = 10, y = 11, z = 12}
// gdbr-check:$12 = vec_slices::AStruct {x: 10, y: 11, z: 12}
// gdb-command:print padded_struct.data_ptr[1]
// gdbg-check:$13 = {x = 13, y = 14, z = 15}
// gdbr-check:$13 = vec_slices::AStruct {x: 13, y: 14, z: 15}

// gdb-command:print mut_slice.length
// gdb-check:$14 = 5
// gdbg-command:print *((i64[5]*)(mut_slice.data_ptr))
// gdbr-command:print *(mut_slice.data_ptr as *const [i64; 5])
// gdbg-check:$15 = {1, 2, 3, 4, 5}
// gdbr-check:$15 = [1, 2, 3, 4, 5]

// Some lines below are marked with [ignored] because old GDB versions seem to have trouble
// accessing globals.

// [ignored] gdbg-command:print 'vec_slices::MUT_VECT_SLICE'.length
// gdbr-command:print MUT_VECT_SLICE.length
// [ignored] gdbg-check:$16 = 2
// gdbr-check:$16 = 2
// [ignored] gdbg-command:print *((i64[2]*)('vec_slices::MUT_VECT_SLICE'.data_ptr))
// gdbr-command:print *(MUT_VECT_SLICE.data_ptr as *const [i64; 2])
// [ignored] gdbg-check:$17 = {64, 65}
// gdbr-check:$17 = [64, 65]

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print empty
// lldbg-check:[...]$0 = size=0
// lldbr-check:(&[i64]) empty = size=0

// lldb-command:print singleton
// lldbg-check:[...]$1 = size=1 { [0] = 1 }
// lldbr-check:(&[i64]) singleton = &[1]

// lldb-command:print multiple
// lldbg-check:[...]$2 = size=4 { [0] = 2 [1] = 3 [2] = 4 [3] = 5 }
// lldbr-check:(&[i64]) multiple = size=4 { [0] = 2 [1] = 3 [2] = 4 [3] = 5 }

// lldb-command:print slice_of_slice
// lldbg-check:[...]$3 = size=2 { [0] = 3 [1] = 4 }
// lldbr-check:(&[i64]) slice_of_slice = size=2 { [0] = 3 [1] = 4 }

// lldb-command:print padded_tuple
// lldbg-check:[...]$4 = size=2 { [0] = { 0 = 6 1 = 7 } [1] = { 0 = 8 1 = 9 } }
// lldbr-check:(&[(i32, i16)]) padded_tuple = size=2 { [0] = { 0 = 6 1 = 7 } [1] = { 0 = 8 1 = 9 } }

// lldb-command:print padded_struct
// lldbg-check:[...]$5 = size=2 { [0] = { x = 10 y = 11 z = 12 } [1] = { x = 13 y = 14 z = 15 } }
// lldbr-check:(&[vec_slices::AStruct]) padded_struct = size=2 { [0] = { x = 10 y = 11 z = 12 } [1] = { x = 13 y = 14 z = 15 } }

#![allow(dead_code, unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct AStruct {
    x: i16,
    y: i32,
    z: i16,
}

static VECT_SLICE: &'static [i64] = &[64, 65];
static mut MUT_VECT_SLICE: &'static [i64] = &[32];

fn main() {
    let empty: &[i64] = &[];
    let singleton: &[i64] = &[1];
    let multiple: &[i64] = &[2, 3, 4, 5];
    let slice_of_slice = &multiple[1..3];

    let padded_tuple: &[(i32, i16)] = &[(6, 7), (8, 9)];

    let padded_struct: &[AStruct] =
        &[AStruct { x: 10, y: 11, z: 12 }, AStruct { x: 13, y: 14, z: 15 }];

    unsafe {
        MUT_VECT_SLICE = VECT_SLICE;
    }

    let mut_slice: &mut [i64] = &mut [1, 2, 3, 4, 5];

    zzz(); // #break
}

fn zzz() {
    ()
}
