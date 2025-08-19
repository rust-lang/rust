//@ ignore-gdb-version: 15.0 - 99.0
// ^ test temporarily disabled as it fails under gdb 15

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print empty.length
// gdb-check:$1 = 0

// gdb-command:print singleton.length
// gdb-check:$2 = 1
// gdb-command:print *(singleton.data_ptr as *const [i64; 1])
// gdb-check:$3 = [1]

// gdb-command:print multiple.length
// gdb-check:$4 = 4
// gdb-command:print *(multiple.data_ptr as *const [i64; 4])
// gdb-check:$5 = [2, 3, 4, 5]

// gdb-command:print slice_of_slice.length
// gdb-check:$6 = 2
// gdb-command:print *(slice_of_slice.data_ptr as *const [i64; 2])
// gdb-check:$7 = [3, 4]

// gdb-command:print padded_tuple.length
// gdb-check:$8 = 2
// gdb-command:print padded_tuple.data_ptr[0]
// gdb-check:$9 = (6, 7)
// gdb-command:print padded_tuple.data_ptr[1]
// gdb-check:$10 = (8, 9)

// gdb-command:print padded_struct.length
// gdb-check:$11 = 2
// gdb-command:print padded_struct.data_ptr[0]
// gdb-check:$12 = vec_slices::AStruct {x: 10, y: 11, z: 12}
// gdb-command:print padded_struct.data_ptr[1]
// gdb-check:$13 = vec_slices::AStruct {x: 13, y: 14, z: 15}

// gdb-command:print mut_slice.length
// gdb-check:$14 = 5
// gdb-command:print *(mut_slice.data_ptr as *const [i64; 5])
// gdb-check:$15 = [1, 2, 3, 4, 5]

// gdb-command:print MUT_VECT_SLICE.length
// gdb-check:$16 = 2
// gdb-command:print *(MUT_VECT_SLICE.data_ptr as *const [i64; 2])
// gdb-check:$17 = [64, 65]

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v empty
// lldb-check:[...] size=0

// lldb-command:v singleton
// lldb-check:[...] size=1 { [0] = 1 }

// lldb-command:v multiple
// lldb-check:[...] size=4 { [0] = 2 [1] = 3 [2] = 4 [3] = 5 }

// lldb-command:v slice_of_slice
// lldb-check:[...] size=2 { [0] = 3 [1] = 4 }

// lldb-command:v padded_tuple
// lldb-check:[...] size=2 { [0] = { 0 = 6 1 = 7 } [1] = { 0 = 8 1 = 9 } }

// lldb-command:v padded_struct
// lldb-check:[...] size=2 { [0] = { x = 10 y = 11 z = 12 } [1] = { x = 13 y = 14 z = 15 } }

#![allow(dead_code, unused_variables)]

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
