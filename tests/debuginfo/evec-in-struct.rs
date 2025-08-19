//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print no_padding1
// gdb-check:$1 = evec_in_struct::NoPadding1 {x: [0, 1, 2], y: -3, z: [4.5, 5.5]}
// gdb-command:print no_padding2
// gdb-check:$2 = evec_in_struct::NoPadding2 {x: [6, 7, 8], y: [[9, 10], [11, 12]]}

// gdb-command:print struct_internal_padding
// gdb-check:$3 = evec_in_struct::StructInternalPadding {x: [13, 14], y: [15, 16]}

// gdb-command:print single_vec
// gdb-check:$4 = evec_in_struct::SingleVec {x: [17, 18, 19, 20, 21]}

// gdb-command:print struct_padded_at_end
// gdb-check:$5 = evec_in_struct::StructPaddedAtEnd {x: [22, 23], y: [24, 25]}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v no_padding1
// lldb-check:[...] { x = { [0] = 0 [1] = 1 [2] = 2 } y = -3 z = { [0] = 4.5 [1] = 5.5 } }
// lldb-command:v no_padding2
// lldb-check:[...] { x = { [0] = 6 [1] = 7 [2] = 8 } y = { [0] = { [0] = 9 [1] = 10 } [1] = { [0] = 11 [1] = 12 } } }

// lldb-command:v struct_internal_padding
// lldb-check:[...] { x = { [0] = 13 [1] = 14 } y = { [0] = 15 [1] = 16 } }

// lldb-command:v single_vec
// lldb-check:[...] { x = { [0] = 17 [1] = 18 [2] = 19 [3] = 20 [4] = 21 } }

// lldb-command:v struct_padded_at_end
// lldb-check:[...] { x = { [0] = 22 [1] = 23 } y = { [0] = 24 [1] = 25 } }

#![allow(unused_variables)]

struct NoPadding1 {
    x: [u32; 3],
    y: i32,
    z: [f32; 2]
}

struct NoPadding2 {
    x: [u32; 3],
    y: [[u32; 2]; 2]
}

struct StructInternalPadding {
    x: [i16; 2],
    y: [i64; 2]
}

struct SingleVec {
    x: [i16; 5]
}

struct StructPaddedAtEnd {
    x: [i64; 2],
    y: [i16; 2]
}

fn main() {

    let no_padding1 = NoPadding1 {
        x: [0, 1, 2],
        y: -3,
        z: [4.5, 5.5]
    };

    let no_padding2 = NoPadding2 {
        x: [6, 7, 8],
        y: [[9, 10], [11, 12]]
    };

    let struct_internal_padding = StructInternalPadding {
        x: [13, 14],
        y: [15, 16]
    };

    let single_vec = SingleVec {
        x: [17, 18, 19, 20, 21]
    };

    let struct_padded_at_end = StructPaddedAtEnd {
        x: [22, 23],
        y: [24, 25]
    };

    zzz(); // #break
}

fn zzz() { () }
