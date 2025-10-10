//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print no_padding1
// gdb-check:$1 = ((0, 1), 2, 3)
// gdb-command:print no_padding2
// gdb-check:$2 = (4, (5, 6), 7)
// gdb-command:print no_padding3
// gdb-check:$3 = (8, 9, (10, 11))

// gdb-command:print internal_padding1
// gdb-check:$4 = (12, (13, 14))
// gdb-command:print internal_padding2
// gdb-check:$5 = (15, (16, 17))

// gdb-command:print padding_at_end1
// gdb-check:$6 = (18, (19, 20))
// gdb-command:print padding_at_end2
// gdb-check:$7 = ((21, 22), 23)


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v no_padding1
// lldb-check:[...] { 0 = { 0 = 0 1 = 1 } 1 = 2 2 = 3 }
// lldb-command:v no_padding2
// lldb-check:[...] { 0 = 4 1 = { 0 = 5 1 = 6 } 2 = 7 }
// lldb-command:v no_padding3
// lldb-check:[...] { 0 = 8 1 = 9 2 = { 0 = 10 1 = 11 } }

// lldb-command:v internal_padding1
// lldb-check:[...] { 0 = 12 1 = { 0 = 13 1 = 14 } }
// lldb-command:v internal_padding2
// lldb-check:[...] { 0 = 15 1 = { 0 = 16 1 = 17 } }

// lldb-command:v padding_at_end1
// lldb-check:[...] { 0 = 18 1 = { 0 = 19 1 = 20 } }
// lldb-command:v padding_at_end2
// lldb-check:[...] { 0 = { 0 = 21 1 = 22 } 1 = 23 }


// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command:dx no_padding1,d
// cdb-check:no_padding1,d [...]: ((0, 1), 2, 3) [Type: tuple$<tuple$<u32,u32>,u32,u32>]
// cdb-check:[...][0]              : (0, 1) [Type: tuple$<u32,u32>]
// cdb-check:[...][1]              : 2 [Type: [...]]
// cdb-check:[...][2]              : 3 [Type: [...]]
// cdb-command:dx no_padding1.__0,d
// cdb-check:no_padding1.__0,d [...]: (0, 1) [Type: tuple$<u32,u32>]
// cdb-check:[...][0]              : 0 [Type: [...]]
// cdb-check:[...][1]              : 1 [Type: [...]]
// cdb-command:dx no_padding2,d
// cdb-check:no_padding2,d [...]: (4, (5, 6), 7) [Type: tuple$<u32,tuple$<u32,u32>,u32>]
// cdb-check:[...][0]              : 4 [Type: [...]]
// cdb-check:[...][1]              : (5, 6) [Type: tuple$<u32,u32>]
// cdb-check:[...][2]              : 7 [Type: [...]]
// cdb-command:dx no_padding2.__1,d
// cdb-check:no_padding2.__1,d [...]: (5, 6) [Type: tuple$<u32,u32>]
// cdb-check:[...][0]              : 5 [Type: [...]]
// cdb-check:[...][1]              : 6 [Type: [...]]
// cdb-command:dx no_padding3,d
// cdb-check:no_padding3,d [...]: (8, 9, (10, 11)) [Type: tuple$<u32,u32,tuple$<u32,u32> >]
// cdb-check:[...][0]              : 8 [Type: [...]]
// cdb-check:[...][1]              : 9 [Type: [...]]
// cdb-check:[...][2]              : (10, 11) [Type: tuple$<u32,u32>]
// cdb-command:dx no_padding3.__2,d
// cdb-check:no_padding3.__2,d [...]: (10, 11) [Type: tuple$<u32,u32>]
// cdb-check:[...][0]              : 10 [Type: [...]]
// cdb-check:[...][1]              : 11 [Type: [...]]

// cdb-command:dx internal_padding1,d
// cdb-check:internal_padding1,d [...]: (12, (13, 14)) [Type: tuple$<i16,tuple$<i32,i32> >]
// cdb-check:[...][0]              : 12 [Type: [...]]
// cdb-check:[...][1]              : (13, 14) [Type: tuple$<i32,i32>]
// cdb-command:dx internal_padding1.__1,d
// cdb-check:internal_padding1.__1,d [...]: (13, 14) [Type: tuple$<i32,i32>]
// cdb-check:[...][0]              : 13 [Type: [...]]
// cdb-check:[...][1]              : 14 [Type: [...]]
// cdb-command:dx internal_padding2,d
// cdb-check:internal_padding2,d [...]: (15, (16, 17)) [Type: tuple$<i16,tuple$<i16,i32> >]
// cdb-check:[...][0]              : 15 [Type: [...]]
// cdb-check:[...][1]              : (16, 17) [Type: tuple$<i16,i32>]
// cdb-command:dx internal_padding2.__1,d
// cdb-check:internal_padding2.__1,d [...]: (16, 17) [Type: tuple$<i16,i32>]
// cdb-check:[...][0]              : 16 [Type: [...]]
// cdb-check:[...][1]              : 17 [Type: [...]]

// cdb-command:dx padding_at_end1,d
// cdb-check:padding_at_end1,d [...]: (18, (19, 20)) [Type: tuple$<i32,tuple$<i32,i16> >]
// cdb-check:[...][0]              : 18 [Type: [...]]
// cdb-check:[...][1]              : (19, 20) [Type: tuple$<i32,i16>]
// cdb-command:dx padding_at_end1.__1,d
// cdb-check:padding_at_end1.__1,d [...][Type: tuple$<i32,i16>]
// cdb-check:[...][0]              : 19 [Type: [...]]
// cdb-check:[...][1]              : 20 [Type: [...]]
// cdb-command:dx padding_at_end2,d
// cdb-check:padding_at_end2,d [...]: ((21, 22), 23) [Type: tuple$<tuple$<i32,i16>,i32>]
// cdb-check:[...][0]              : (21, 22) [Type: tuple$<i32,i16>]
// cdb-check:[...][1]              : 23 [Type: [...]]
// cdb-command:dx padding_at_end2.__0,d
// cdb-check:padding_at_end2.__0,d [...]: (21, 22) [Type: tuple$<i32,i16>]
// cdb-check:[...][0]              : 21 [Type: [...]]
// cdb-check:[...][1]              : 22 [Type: [...]]

#![allow(unused_variables)]

fn main() {
    let no_padding1: ((u32, u32), u32, u32) = ((0, 1), 2, 3);
    let no_padding2: (u32, (u32, u32), u32) = (4, (5, 6), 7);
    let no_padding3: (u32, u32, (u32, u32)) = (8, 9, (10, 11));

    let internal_padding1: (i16, (i32, i32)) = (12, (13, 14));
    let internal_padding2: (i16, (i16, i32)) = (15, (16, 17));

    let padding_at_end1: (i32, (i32, i16)) = (18, (19, 20));
    let padding_at_end2: ((i32, i16), i32) = ((21, 22), 23);

    zzz(); // #break
}

fn zzz() {()}
