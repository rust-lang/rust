#![deny(unused_tuple_struct_fields)]
//~^ NOTE: the lint level is defined here

use std::marker::PhantomData;

const LEN: usize = 4;

struct SingleUnused(i32, [u8; LEN], String);
//~^ ERROR: field `1` is never read
//~| NOTE: field in this struct
//~| HELP: consider changing the field to be of unit type

struct MultipleUnused(i32, f32, String, u8);
//~^ ERROR: fields `0`, `1`, `2`, and `3` are never read
//~| NOTE: fields in this struct
//~| HELP: consider changing the fields to be of unit type

struct GoodUnit(());

struct GoodPhantom(PhantomData<i32>);

struct Void;
struct GoodVoid(Void);

fn main() {
    let w = SingleUnused(42, [0, 1, 2, 3], "abc".to_string());
    let _ = w.0;
    let _ = w.2;

    let m = MultipleUnused(42, 3.14, "def".to_string(), 4u8);

    let gu = GoodUnit(());
    let gp = GoodPhantom(PhantomData);
    let gv = GoodVoid(Void);

    let _ = (gu, gp, gv, m);
}
