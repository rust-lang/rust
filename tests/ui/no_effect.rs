#![feature(plugin, box_syntax, inclusive_range_syntax)]
#![plugin(clippy)]

#![deny(no_effect, unnecessary_operation)]
#![allow(dead_code)]
#![allow(path_statements)]
#![allow(deref_addrof)]
#![feature(untagged_unions)]

struct Unit;
struct Tuple(i32);
struct Struct {
    field: i32
}
enum Enum {
    Tuple(i32),
    Struct { field: i32 },
}

union Union {
    a: u8,
    b: f64,
}

fn get_number() -> i32 { 0 }
fn get_struct() -> Struct { Struct { field: 0 } }

unsafe fn unsafe_fn() -> i32 { 0 }

fn main() {
    let s = get_struct();
    let s2 = get_struct();

    0;
    s2;
    Unit;
    Tuple(0);
    Struct { field: 0 };
    Struct { ..s };
    Union { a: 0 };
    Enum::Tuple(0);
    Enum::Struct { field: 0 };
    5 + 6;
    *&42;
    &6;
    (5, 6, 7);
    box 42;
    ..;
    5..;
    ..5;
    5..6;
    5...6;
    [42, 55];
    [42, 55][1];
    (42, 55).1;
    [42; 55];
    [42; 55][13];
    let mut x = 0;
    || x += 5;

    // Do not warn
    get_number();
    unsafe { unsafe_fn() };

    Tuple(get_number());
    Struct { field: get_number() };
    Struct { ..get_struct() };
    Enum::Tuple(get_number());
    Enum::Struct { field: get_number() };
    5 + get_number();
    *&get_number();
    &get_number();
    (5, 6, get_number());
    box get_number();
    get_number()..;
    ..get_number();
    5..get_number();
    [42, get_number()];
    [42, 55][get_number() as usize];
    (42, get_number()).1;
    [get_number(); 55];
    [42; 55][get_number() as usize];
    {get_number()};
}
