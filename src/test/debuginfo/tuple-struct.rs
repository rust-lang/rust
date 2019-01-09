// ignore-tidy-linelength

// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print no_padding16
// gdbg-check:$1 = {__0 = 10000, __1 = -10001}
// gdbr-check:$1 = tuple_struct::NoPadding16 (10000, -10001)

// gdb-command:print no_padding32
// gdbg-check:$2 = {__0 = -10002, __1 = -10003.5, __2 = 10004}
// gdbr-check:$2 = tuple_struct::NoPadding32 (-10002, -10003.5, 10004)

// gdb-command:print no_padding64
// gdbg-check:$3 = {__0 = -10005.5, __1 = 10006, __2 = 10007}
// gdbr-check:$3 = tuple_struct::NoPadding64 (-10005.5, 10006, 10007)

// gdb-command:print no_padding163264
// gdbg-check:$4 = {__0 = -10008, __1 = 10009, __2 = 10010, __3 = 10011}
// gdbr-check:$4 = tuple_struct::NoPadding163264 (-10008, 10009, 10010, 10011)

// gdb-command:print internal_padding
// gdbg-check:$5 = {__0 = 10012, __1 = -10013}
// gdbr-check:$5 = tuple_struct::InternalPadding (10012, -10013)

// gdb-command:print padding_at_end
// gdbg-check:$6 = {__0 = -10014, __1 = 10015}
// gdbr-check:$6 = tuple_struct::PaddingAtEnd (-10014, 10015)


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print no_padding16
// lldbg-check:[...]$0 = NoPadding16(10000, -10001)
// lldbr-check:(tuple_struct::NoPadding16) no_padding16 = { = 10000 = -10001 }

// lldb-command:print no_padding32
// lldbg-check:[...]$1 = NoPadding32(-10002, -10003.5, 10004)
// lldbr-check:(tuple_struct::NoPadding32) no_padding32 = { = -10002 = -10003.5 = 10004 }

// lldb-command:print no_padding64
// lldbg-check:[...]$2 = NoPadding64(-10005.5, 10006, 10007)
// lldbr-check:(tuple_struct::NoPadding64) no_padding64 = { = -10005.5 = 10006 = 10007 }

// lldb-command:print no_padding163264
// lldbg-check:[...]$3 = NoPadding163264(-10008, 10009, 10010, 10011)
// lldbr-check:(tuple_struct::NoPadding163264) no_padding163264 = { = -10008 = 10009 = 10010 = 10011 }

// lldb-command:print internal_padding
// lldbg-check:[...]$4 = InternalPadding(10012, -10013)
// lldbr-check:(tuple_struct::InternalPadding) internal_padding = { = 10012 = -10013 }

// lldb-command:print padding_at_end
// lldbg-check:[...]$5 = PaddingAtEnd(-10014, 10015)
// lldbr-check:(tuple_struct::PaddingAtEnd) padding_at_end = { = -10014 = 10015 }

// This test case mainly makes sure that no field names are generated for tuple structs (as opposed
// to all fields having the name "<unnamed_field>"). Otherwise they are handled the same a normal
// structs.


#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct NoPadding16(u16, i16);
struct NoPadding32(i32, f32, u32);
struct NoPadding64(f64, i64, u64);
struct NoPadding163264(i16, u16, i32, u64);
struct InternalPadding(u16, i64);
struct PaddingAtEnd(i64, u16);

fn main() {
    let no_padding16 = NoPadding16(10000, -10001);
    let no_padding32 = NoPadding32(-10002, -10003.5, 10004);
    let no_padding64 = NoPadding64(-10005.5, 10006, 10007);
    let no_padding163264 = NoPadding163264(-10008, 10009, 10010, 10011);

    let internal_padding = InternalPadding(10012, -10013);
    let padding_at_end = PaddingAtEnd(-10014, 10015);

    zzz(); // #break
}

fn zzz() {()}
