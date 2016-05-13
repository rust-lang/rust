#![feature(plugin, box_syntax, inclusive_range_syntax)]
#![plugin(clippy)]

#![deny(no_effect, unnecessary_operation)]
#![allow(dead_code)]
#![allow(path_statements)]

struct Unit;
struct Tuple(i32);
struct Struct {
    field: i32
}
enum Enum {
    Tuple(i32),
    Struct { field: i32 },
}

fn get_number() -> i32 { 0 }
fn get_struct() -> Struct { Struct { field: 0 } }

fn main() {
    let s = get_struct();
    let s2 = get_struct();

    0; //~ERROR statement with no effect
    s2; //~ERROR statement with no effect
    Unit; //~ERROR statement with no effect
    Tuple(0); //~ERROR statement with no effect
    Struct { field: 0 }; //~ERROR statement with no effect
    Struct { ..s }; //~ERROR statement with no effect
    Enum::Tuple(0); //~ERROR statement with no effect
    Enum::Struct { field: 0 }; //~ERROR statement with no effect
    5 + 6; //~ERROR statement with no effect
    *&42; //~ERROR statement with no effect
    &6; //~ERROR statement with no effect
    (5, 6, 7); //~ERROR statement with no effect
    box 42; //~ERROR statement with no effect
    ..; //~ERROR statement with no effect
    5..; //~ERROR statement with no effect
    ..5; //~ERROR statement with no effect
    5..6; //~ERROR statement with no effect
    5...6; //~ERROR statement with no effect
    [42, 55]; //~ERROR statement with no effect
    [42, 55][1]; //~ERROR statement with no effect
    (42, 55).1; //~ERROR statement with no effect
    [42; 55]; //~ERROR statement with no effect
    [42; 55][13]; //~ERROR statement with no effect
    let mut x = 0;
    || x += 5; //~ERROR statement with no effect

    // Do not warn
    get_number();

    Tuple(get_number()); //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION get_number();
    Struct { field: get_number() }; //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION get_number();
    Struct { ..get_struct() }; //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION get_number();
    Enum::Tuple(get_number()); //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION get_number();
    Enum::Struct { field: get_number() }; //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION get_number();
    5 + get_number(); //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION 5;get_number();
    *&get_number(); //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION &get_number();
    &get_number(); //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION get_number();
    (5, 6, get_number()); //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION 5;6;get_number();
    box get_number(); //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION get_number();
    get_number()..; //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION get_number();
    ..get_number(); //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION get_number();
    5..get_number(); //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION 5;get_number();
    [42, get_number()]; //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION 42;get_number();
    [42, 55][get_number() as usize]; //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION [42, 55];get_number() as usize;
    (42, get_number()).1; //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION 42;get_number();
    [get_number(); 55]; //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION get_number();
    [42; 55][get_number() as usize]; //~ERROR statement can be reduced
    //~^HELP replace it with
    //~|SUGGESTION [42; 55];get_number() as usize;
}
