//@ min-lldb-version: 1800
//@ min-gdb-version: 13.0

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print some
// gdb-check:$1 = core::option::Option<&u32>::Some(0x[...])

// gdb-command:print none
// gdb-check:$2 = core::option::Option<&u32>::None

// gdb-command:print full
// gdb-check:$3 = option_like_enum::MoreFields::Full(454545, 0x[...], 9988)

// gdb-command:print empty
// gdb-check:$4 = option_like_enum::MoreFields::Empty

// gdb-command:print droid
// gdb-check:$5 = option_like_enum::NamedFields::Droid{id: 675675, range: 10000001, internals: 0x[...]}

// gdb-command:print void_droid
// gdb-check:$6 = option_like_enum::NamedFields::Void

// gdb-command:print nested_non_zero_yep
// gdb-check:$7 = option_like_enum::NestedNonZero::Yep(10.5, option_like_enum::NestedNonZeroField {a: 10, b: 20, c: 0x[...]})

// gdb-command:print nested_non_zero_nope
// gdb-check:$8 = option_like_enum::NestedNonZero::Nope

// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v some
// lldb-check:(core::option::Option<&u32>) some = { value = { 0 = 0x0000555555556014 } }

// lldb-command:v none
// lldb-check:(core::option::Option<&u32>) none = { value = $discr$ = 0 }

// lldb-command:v full
// lldb-check:(option_like_enum::MoreFields) full = { value = { 0 = 454545 1 = 0x0000555555556008 2 = 9988 } $discr$ = 0 }

// lldb-command:v empty
// lldb-check:(option_like_enum::MoreFields) empty = { value = $discr$ = 1 }

// lldb-command:v droid
// lldb-check:(option_like_enum::NamedFields) droid = { value = { id = 675675 range = 10000001 internals = 0x0000555555556008 } $discr$ = 0 }

// lldb-command:v void_droid
// lldb-check:(option_like_enum::NamedFields) void_droid = { value = $discr$ = 1 }

// lldb-command:v some_str
// lldb-check:(core::option::Option<&str>) some_str = { value = { 0 = "abc" { [0] = 'a' [1] = 'b' [2] = 'c' } } }

// lldb-command:v none_str
// lldb-check:(core::option::Option<&str>) none_str = { value = $discr$ = 0 }

// lldb-command:v nested_non_zero_yep
// lldb-check:(option_like_enum::NestedNonZero) nested_non_zero_yep = { value = { 0 = 10.5 1 = { a = 10 b = 20 c = 0x0000555555556018 } } }

// lldb-command:v nested_non_zero_nope
// lldb-check:nested_non_zero_nope = { value = $discr$ = 0 }


// If a struct has exactly two variants, one of them is empty, and the other one
// contains a non-nullable pointer, then this value is used as the discriminator.
// The test cases in this file make sure that something readable is generated for
// this kind of types.

enum MoreFields<'a> {
    Full(u32, &'a isize, i16),
    Empty
}

struct MoreFieldsRepr<'a> {
    a: u32,
    discr: &'a isize,
    b: i16
}

enum NamedFields<'a> {
    Droid { id: i32, range: i64, internals: &'a isize },
    Void
}

struct NamedFieldsRepr<'a> {
    id: i32,
    range: i64,
    internals: &'a isize
}

struct NestedNonZeroField<'a> {
    a: u16,
    b: u32,
    c: &'a char,
}

enum NestedNonZero<'a> {
    Yep(f64, NestedNonZeroField<'a>),
    Nope
}

fn main() {

    let some_str: Option<&'static str> = Some("abc");
    let none_str: Option<&'static str> = None;

    let some: Option<&u32> = Some(&1234);
    let none: Option<&u32> = None;

    let full = MoreFields::Full(454545, &1234, 9988);
    let empty = MoreFields::Empty;

    let droid = NamedFields::Droid {
        id: 675675,
        range: 10000001,
        internals: &1234,
    };
    let void_droid = NamedFields::Void;

    let nested_non_zero_yep = NestedNonZero::Yep(
        10.5,
        NestedNonZeroField {
            a: 10,
            b: 20,
            c: &'x',
        });
    let nested_non_zero_nope = NestedNonZero::Nope;

    zzz(); // #break
}

fn zzz() {()}
