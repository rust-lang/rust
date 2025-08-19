//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print no_padding16
// gdb-check:$1 = tuple_struct::NoPadding16 (10000, -10001)

// gdb-command:print no_padding32
// gdb-check:$2 = tuple_struct::NoPadding32 (-10002, -10003.5, 10004)

// gdb-command:print no_padding64
// gdb-check:$3 = tuple_struct::NoPadding64 (-10005.5, 10006, 10007)

// gdb-command:print no_padding163264
// gdb-check:$4 = tuple_struct::NoPadding163264 (-10008, 10009, 10010, 10011)

// gdb-command:print internal_padding
// gdb-check:$5 = tuple_struct::InternalPadding (10012, -10013)

// gdb-command:print padding_at_end
// gdb-check:$6 = tuple_struct::PaddingAtEnd (-10014, 10015)


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v no_padding16
// lldb-check:[...] { 0 = 10000 1 = -10001 }

// lldb-command:v no_padding32
// lldb-check:[...] { 0 = -10002 1 = -10003.5 2 = 10004 }

// lldb-command:v no_padding64
// lldb-check:[...] { 0 = -10005.5 1 = 10006 2 = 10007 }

// lldb-command:v no_padding163264
// lldb-check:[...] { 0 = -10008 1 = 10009 2 = 10010 3 = 10011 }

// lldb-command:v internal_padding
// lldb-check:[...] { 0 = 10012 1 = -10013 }

// lldb-command:v padding_at_end
// lldb-check:[...] { 0 = -10014 1 = 10015 }

// This test case mainly makes sure that no field names are generated for tuple structs (as opposed
// to all fields having the name "<unnamed_field>"). Otherwise they are handled the same a normal
// structs.


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
