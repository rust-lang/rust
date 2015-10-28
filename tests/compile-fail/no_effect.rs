#![feature(plugin)]
#![plugin(clippy)]

#![deny(no_effect)]
#![allow(dead_code)]
#![allow(path_statements)]

struct Unit;
struct Tuple(i32);
struct Struct {
    field: i32
}
enum Enum {
    TupleVariant(i32),
    StructVariant { field: i32 },
}

fn get_number() -> i32 { 0 }
fn get_struct() -> Struct { Struct { field: 0 } }

fn main() {
    let s = get_struct();

    0; //~ERROR statement with no effect
    Unit; //~ERROR statement with no effect
    Tuple(0); //~ERROR statement with no effect
    Struct { field: 0 }; //~ERROR statement with no effect
    Struct { ..s }; //~ERROR statement with no effect
    Enum::TupleVariant(0); //~ERROR statement with no effect
    Enum::StructVariant { field: 0 }; //~ERROR statement with no effect

    // Do not warn
    get_number();
    Tuple(get_number());
    Struct { field: get_number() };
    Struct { ..get_struct() };
    Enum::TupleVariant(get_number());
    Enum::StructVariant { field: get_number() };
}
