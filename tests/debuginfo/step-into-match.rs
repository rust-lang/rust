//@ compile-flags: -g
//@ ignore-android: FIXME(#10381)

// On Arm64 Windows, stepping at the end of a function on goes to the callsite, not the instruction
// after it.
//@ ignore-aarch64-pc-windows-msvc: Stepping out of functions behaves differently.

// === GDB TESTS ==============================================================

// gdb-command: r

// gdb-command: s
// gdb-check:[...]match x {

// gdb-command: s
// gdb-check:[...]    Some(42) => 1,

// gdb-command: s
// gdb-check:[...]}

// gdb-command: s
// gdb-check:[...]match_enum(Some(12));

// gdb-command: s
// gdb-check:[...]match x {

// gdb-command: s
// gdb-check:[...]Some(_) => 2,

// gdb-command: s
// gdb-check:[...]}

// gdb-command: s
// gdb-check:[...]match_enum(None);

// gdb-command: s
// gdb-check:[...]match x {

// gdb-command: s
// gdb-check:[...]None => 3,

// gdb-command: s
// gdb-check:[...]}

// gdb-command: s
// gdb-check:[...]match_int(1);

// gdb-command: s
// gdb-check:[...]match y {

// gdb-command: s
// gdb-check:[...]1 => 3,

// gdb-command: s
// gdb-check:[...]}

// gdb-command: s
// gdb-check:[...]match_int(2);

// gdb-command: s
// gdb-check:[...]match y {

// gdb-command: s
// gdb-check:[...]_ => 4,

// gdb-command: s
// gdb-check:[...]}

// gdb-command: s
// gdb-check:[...]match_int(0);

// gdb-command: s
// gdb-check:[...]match y {

// gdb-command: s
// gdb-check:[...]0 => 2,

// gdb-command: s
// gdb-check:[...]}

// gdb-command: s
// gdb-check:[...]match_int(-1);

// gdb-command: s
// gdb-check:[...]match y {

// gdb-command: s
// gdb-check:[...]-1 => 1,

// gdb-command: s
// gdb-check:[...]}

// gdb-command: s
// gdb-check:[...]match_tuple(5, 12);

// gdb-command: s
// gdb-check:[...]match (a, b) {

// gdb-command: s
// gdb-check:[...](5, 12) => 3,

// gdb-command: s
// gdb-check:[...]}

// gdb-command: s
// gdb-check:[...]match_tuple(29, 1);

// gdb-command: s
// gdb-check:[...]match (a, b) {

// gdb-command: s
// gdb-check:[...](29, _) => 2,

// gdb-command: s
// gdb-check:[...]}

// gdb-command: s
// gdb-check:[...]match_tuple(12, 12);

// gdb-command: s
// gdb-check:[...]match (a, b) {

// gdb-command: s
// gdb-check:[...](_, _) => 5,

// gdb-command: s
// gdb-check:[...]}

// gdb-command: s
// gdb-check:[...]match_tuple(42, 12);

// gdb-command: s
// gdb-check:[...]match (a, b) {

// gdb-command: s
// gdb-check:[...](42, 12) => 1,

// gdb-command: s
// gdb-check:[...]}

// gdb-command: s
// gdb-check:[...]match_tuple(1, 9);

// gdb-command: s
// gdb-check:[...]match (a, b) {

// gdb-command: s
// gdb-check:[...](_, 9) => 4,

// gdb-command: s
// gdb-check:[...]}

// gdb-command: s
// gdb-check:[...]}

// === CDB TESTS ==============================================================

// Enable line-based debugging and print lines after stepping.
// cdb-command: .lines -e
// cdb-command: l+s
// cdb-command: l+t

// cdb-command: g

// cdb-command: t
// cdb-check:   [...]: fn match_enum(x: Option<u32>) -> u8 {

// cdb-command: t
// cdb-check:   [...]:     match x {

// cdb-command: t
// cdb-check:   [...]:         Some(42) => 1,

// cdb-command: t
// cdb-check:   [...]: }

// cdb-command: t
// cdb-check:   [...]:     match_enum(Some(12));

// cdb-command: t
// cdb-check:   [...]: fn match_enum(x: Option<u32>) -> u8 {

// cdb-command: t
// cdb-check:   [...]:     match x {

// cdb-command: t
// cdb-check:   [...]:         Some(_) => 2,

// cdb-command: t
// cdb-check:   [...]: }

// cdb-command: t
// cdb-check:   [...]:     match_enum(None);

// cdb-command: t
// cdb-check:   [...]: fn match_enum(x: Option<u32>) -> u8 {

// cdb-command: t
// cdb-check:   [...]:     match x {

// cdb-command: t
// cdb-check:   [...]:         None => 3,

// cdb-command: t
// cdb-check:   [...]: }

// cdb-command: t
// cdb-check:   [...]:     match_int(1);

// cdb-command: t
// cdb-check:   [...]: fn match_int(y: i32) -> u16 {

// cdb-command: t
// cdb-check:   [...]:     match y {

// cdb-command: t
// cdb-check:   [...]:         1 => 3,

// cdb-command: t
// cdb-check:   [...]: }

// cdb-command: t
// cdb-check:   [...]:     match_int(2);

// cdb-command: t
// cdb-check:   [...]: fn match_int(y: i32) -> u16 {

// cdb-command: t
// cdb-check:   [...]:     match y {

// cdb-command: t
// cdb-check:   [...]:         _ => 4,

// cdb-command: t
// cdb-check:   [...]: }

// cdb-command: t
// cdb-check:   [...]:     match_int(0);

// cdb-command: t
// cdb-check:   [...]: fn match_int(y: i32) -> u16 {

// cdb-command: t
// cdb-check:   [...]:     match y {

// cdb-command: t
// cdb-check:   [...]:         0 => 2,

// cdb-command: t
// cdb-check:   [...]: }

// cdb-command: t
// cdb-check:   [...]:     match_int(-1);

// cdb-command: t
// cdb-check:   [...]: fn match_int(y: i32) -> u16 {

// cdb-command: t
// cdb-check:   [...]:     match y {

// cdb-command: t
// cdb-check:   [...]:         -1 => 1,

// cdb-command: t
// cdb-check:   [...]: }

// cdb-command: t
// cdb-check:   [...]:     match_tuple(5, 12);

// cdb-command: t
// cdb-check:   [...]: fn match_tuple(a: u8, b: i8) -> u32 {

// cdb-command: t
// cdb-check:   [...]:     match (a, b) {

// cdb-command: t
// cdb-check:   [...]:         (5, 12) => 3,

// cdb-command: t
// cdb-check:   [...]: }

// cdb-command: t
// cdb-check:   [...]:     match_tuple(29, 1);

// cdb-command: t
// cdb-check:   [...]: fn match_tuple(a: u8, b: i8) -> u32 {

// cdb-command: t
// cdb-check:   [...]:     match (a, b) {

// cdb-command: t
// cdb-check:   [...]:         (29, _) => 2,

// cdb-command: t
// cdb-check:   [...]: }

// cdb-command: t
// cdb-check:   [...]:     match_tuple(12, 12);

// cdb-command: t
// cdb-check:   [...]: fn match_tuple(a: u8, b: i8) -> u32 {

// cdb-command: t
// cdb-check:   [...]:     match (a, b) {

// cdb-command: t
// cdb-check:   [...]:         (_, _) => 5,

// cdb-command: t
// cdb-check:   [...]: }

// cdb-command: t
// cdb-check:   [...]:     match_tuple(42, 12);

// cdb-command: t
// cdb-check:   [...]: fn match_tuple(a: u8, b: i8) -> u32 {

// cdb-command: t
// cdb-check:   [...]:     match (a, b) {

// cdb-command: t
// cdb-check:   [...]:         (42, 12) => 1,

// cdb-command: t
// cdb-check:   [...]: }

// cdb-command: t
// cdb-check:   [...]:     match_tuple(1, 9);

// cdb-command: t
// cdb-check:   [...]: fn match_tuple(a: u8, b: i8) -> u32 {

// cdb-command: t
// cdb-check:   [...]:     match (a, b) {

// cdb-command: t
// cdb-check:   [...]:         (_, 9) => 4,

// cdb-command: t
// cdb-check:   [...]: }

// cdb-command: t
// cdb-check:   [...]: }

fn main() {
    match_enum(Some(42)); // #break
    match_enum(Some(12));
    match_enum(None);

    match_int(1);
    match_int(2);
    match_int(0);
    match_int(-1);

    match_tuple(5, 12);
    match_tuple(29, 1);
    match_tuple(12, 12);
    match_tuple(42, 12);
    match_tuple(1, 9);
}

fn match_enum(x: Option<u32>) -> u8 {
    match x {
        Some(42) => 1,
        Some(_) => 2,
        None => 3,
    }
}

fn match_int(y: i32) -> u16 {
    match y {
        -1 => 1,
        0 => 2,
        1 => 3,
        _ => 4,
    }
}

fn match_tuple(a: u8, b: i8) -> u32 {
    match (a, b) {
        (42, 12) => 1,
        (29, _) => 2,
        (5, 12) => 3,
        (_, 9) => 4,
        (_, _) => 5,
    }
}
