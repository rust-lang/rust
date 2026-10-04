#![deny(dead_code)]
//~^ NOTE: the lint level is defined here

use std::marker::PhantomData;

const LEN: usize = 4;

struct UnusedAtTheEnd(i32, f32, [u8; LEN], String, u8);
//~^ ERROR:fields `1`, `2`, `3`, and `4` are never read
//~| NOTE: fields in this struct
//~| HELP: consider removing these fields

struct UnusedJustOneField(i32);
//~^ ERROR: field `0` is never read
//~| NOTE: field in this struct
//~| HELP: consider removing this field

struct UnusedInTheMiddle(i32, f32, String, u8, u32);
//~^ ERROR: fields `1`, `2`, and `4` are never read
//~| NOTE: fields in this struct
//~| HELP: consider changing the fields to be of unit type to suppress this warning while preserving the field numbering, or remove the fields

struct GoodUnit(());

struct GoodPhantom(PhantomData<i32>);

struct Void;
struct GoodVoid(Void);

fn main() {
    let u1 = UnusedAtTheEnd(42, 3.14, [0, 1, 2, 3], "def".to_string(), 4u8);
    let _ = u1.0;

    let _ = UnusedJustOneField(42);

    let u2 = UnusedInTheMiddle(42, 3.14, "def".to_string(), 4u8, 5);
    let _ = u2.0;
    let _ = u2.3;


    let gu = GoodUnit(());
    let gp = GoodPhantom(PhantomData);
    let gv = GoodVoid(Void);

    let _ = (gu, gp, gv);
}
