#![feature(plugin, box_syntax, inclusive_range_syntax)]


#![warn(no_effect, unnecessary_operation)]
#![allow(dead_code)]
#![allow(path_statements)]
#![allow(deref_addrof)]
#![allow(redundant_field_names)]
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
struct DropUnit;
impl Drop for DropUnit {
    fn drop(&mut self) {}
}
struct DropStruct {
    field: i32
}
impl Drop for DropStruct {
    fn drop(&mut self) {}
}
struct DropTuple(i32);
impl Drop for DropTuple {
    fn drop(&mut self) {}
}
enum DropEnum {
    Tuple(i32),
    Struct { field: i32 },
}
impl Drop for DropEnum {
    fn drop(&mut self) {}
}
struct FooString {
    s: String,
}
union Union {
    a: u8,
    b: f64,
}

fn get_number() -> i32 { 0 }
fn get_struct() -> Struct { Struct { field: 0 } }
fn get_drop_struct() -> DropStruct { DropStruct { field: 0 } }

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
    5..=6;
    [42, 55];
    [42, 55][1];
    (42, 55).1;
    [42; 55];
    [42; 55][13];
    let mut x = 0;
    || x += 5;
    let s: String = "foo".into();
    FooString { s: s };

    // Do not warn
    get_number();
    unsafe { unsafe_fn() };
    DropUnit;
    DropStruct { field: 0 };
    DropTuple(0);
    DropEnum::Tuple(0);
    DropEnum::Struct { field: 0 };

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
    FooString { s: String::from("blah"), };

    // Do not warn
    DropTuple(get_number());
    DropStruct { field: get_number() };
    DropStruct { field: get_number() };
    DropStruct { ..get_drop_struct() };
    DropEnum::Tuple(get_number());
    DropEnum::Struct { field: get_number() };
}
