//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:print simple_tuple::NO_PADDING_8
// gdb-check:$1 = (-50, 50)
// gdb-command:print simple_tuple::NO_PADDING_16
// gdb-check:$2 = (-1, 2, 3)
// gdb-command:print simple_tuple::NO_PADDING_32
// gdb-check:$3 = (4, 5, 6)
// gdb-command:print simple_tuple::NO_PADDING_64
// gdb-check:$4 = (7, 8, 9)

// gdb-command:print simple_tuple::INTERNAL_PADDING_1
// gdb-check:$5 = (10, 11)
// gdb-command:print simple_tuple::INTERNAL_PADDING_2
// gdb-check:$6 = (12, 13, 14, 15)

// gdb-command:print simple_tuple::PADDING_AT_END
// gdb-check:$7 = (16, 17)

// gdb-command:run

// gdb-command:print noPadding8
// gdb-check:$8 = (-100, 100)
// gdb-command:print noPadding16
// gdb-check:$9 = (0, 1, 2)
// gdb-command:print noPadding32
// gdb-check:$10 = (3, 4.5, 5)
// gdb-command:print noPadding64
// gdb-check:$11 = (6, 7.5, 8)

// gdb-command:print internalPadding1
// gdb-check:$12 = (9, 10)
// gdb-command:print internalPadding2
// gdb-check:$13 = (11, 12, 13, 14)

// gdb-command:print paddingAtEnd
// gdb-check:$14 = (15, 16)

// gdb-command:print simple_tuple::NO_PADDING_8
// gdb-check:$15 = (-127, 127)
// gdb-command:print simple_tuple::NO_PADDING_16
// gdb-check:$16 = (-10, 10, 9)
// gdb-command:print simple_tuple::NO_PADDING_32
// gdb-check:$17 = (14, 15, 16)
// gdb-command:print simple_tuple::NO_PADDING_64
// gdb-check:$18 = (17, 18, 19)

// gdb-command:print simple_tuple::INTERNAL_PADDING_1
// gdb-check:$19 = (110, 111)
// gdb-command:print simple_tuple::INTERNAL_PADDING_2
// gdb-check:$20 = (112, 113, 114, 115)

// gdb-command:print simple_tuple::PADDING_AT_END
// gdb-check:$21 = (116, 117)


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v/d noPadding8
// lldb-check:[...] { 0 = -100 1 = 100 }
// lldb-command:v noPadding16
// lldb-check:[...] { 0 = 0 1 = 1 2 = 2 }
// lldb-command:v noPadding32
// lldb-check:[...] { 0 = 3 1 = 4.5 2 = 5 }
// lldb-command:v noPadding64
// lldb-check:[...] { 0 = 6 1 = 7.5 2 = 8 }

// lldb-command:v internalPadding1
// lldb-check:[...] { 0 = 9 1 = 10 }
// lldb-command:v internalPadding2
// lldb-check:[...] { 0 = 11 1 = 12 2 = 13 3 = 14 }

// lldb-command:v paddingAtEnd
// lldb-check:[...] { 0 = 15 1 = 16 }


// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command:dx noPadding8,d
// cdb-check:noPadding8,d [...]: (-100, 100) [Type: tuple$<i8,u8>]
// cdb-check:[...][0]              : -100 [Type: [...]]
// cdb-check:[...][1]              : 100 [Type: [...]]
// cdb-command:dx noPadding16,d
// cdb-check:noPadding16,d [...]: (0, 1, 2) [Type: tuple$<i16,i16,u16>]
// cdb-check:[...][0]              : 0 [Type: [...]]
// cdb-check:[...][1]              : 1 [Type: [...]]
// cdb-check:[...][2]              : 2 [Type: [...]]
// cdb-command:dx noPadding32,d
// cdb-check:noPadding32,d [...]: (3, 4.5[...], 5) [Type: tuple$<i32,f32,u32>]
// cdb-check:[...][0]              : 3 [Type: [...]]
// cdb-check:[...][1]              : 4.5[...] [Type: [...]]
// cdb-check:[...][2]              : 5 [Type: [...]]
// cdb-command:dx noPadding64,d
// cdb-check:noPadding64,d [...]: (6, 7.5[...], 8) [Type: tuple$<i64,f64,u64>]
// cdb-check:[...][0]              : 6 [Type: [...]]
// cdb-check:[...][1]              : 7.500000 [Type: [...]]
// cdb-check:[...][2]              : 8 [Type: [...]]

// cdb-command:dx internalPadding1,d
// cdb-check:internalPadding1,d [...]: (9, 10) [Type: tuple$<i16,i32>]
// cdb-check:[...][0]              : 9 [Type: short]
// cdb-check:[...][1]              : 10 [Type: int]
// cdb-command:dx internalPadding2,d
// cdb-check:internalPadding2,d [...]: (11, 12, 13, 14) [Type: tuple$<i16,i32,u32,u64>]
// cdb-check:[...][0]              : 11 [Type: [...]]
// cdb-check:[...][1]              : 12 [Type: [...]]
// cdb-check:[...][2]              : 13 [Type: [...]]
// cdb-check:[...][3]              : 14 [Type: [...]]

// cdb-command:dx paddingAtEnd,d
// cdb-check:paddingAtEnd,d [...]: (15, 16) [Type: tuple$<i32,i16>]
// cdb-check:[...][0]              : 15 [Type: [...]]
// cdb-check:[...][1]              : 16 [Type: [...]]


#![allow(unused_variables)]
#![allow(dead_code)]

static mut NO_PADDING_8: (i8, u8) = (-50, 50);
static mut NO_PADDING_16: (i16, i16, u16) = (-1, 2, 3);

static mut NO_PADDING_32: (i32, f32, u32) = (4, 5.0, 6);
static mut NO_PADDING_64: (i64, f64, u64) = (7, 8.0, 9);

static mut INTERNAL_PADDING_1: (i16, i32) = (10, 11);
static mut INTERNAL_PADDING_2: (i16, i32, u32, u64) = (12, 13, 14, 15);

static mut PADDING_AT_END: (i32, i16) = (16, 17);

fn main() {
    let noPadding8: (i8, u8) = (-100, 100);
    let noPadding16: (i16, i16, u16) = (0, 1, 2);
    let noPadding32: (i32, f32, u32) = (3, 4.5, 5);
    let noPadding64: (i64, f64, u64) = (6, 7.5, 8);

    let internalPadding1: (i16, i32) = (9, 10);
    let internalPadding2: (i16, i32, u32, u64) = (11, 12, 13, 14);

    let paddingAtEnd: (i32, i16) = (15, 16);

    unsafe {
        NO_PADDING_8 = (-127, 127);
        NO_PADDING_16 = (-10, 10, 9);

        NO_PADDING_32 = (14, 15.0, 16);
        NO_PADDING_64 = (17, 18.0, 19);

        INTERNAL_PADDING_1 = (110, 111);
        INTERNAL_PADDING_2 = (112, 113, 114, 115);

        PADDING_AT_END = (116, 117);
    }

    zzz(); // #break
}

fn zzz() {()}
