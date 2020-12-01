#![allow(unused)]
#![warn(clippy::ref_option_ref)]

// This lint is not tagged as run-rustfix because automatically
// changing the type of a variable would also means changing
// all usages of this variable to match and This is not handled
// by this lint.

static THRESHOLD: i32 = 10;
static REF_THRESHOLD: &Option<&i32> = &Some(&THRESHOLD);
const CONST_THRESHOLD: &i32 = &10;
const REF_CONST: &Option<&i32> = &Some(&CONST_THRESHOLD);

type RefOptRefU32<'a> = &'a Option<&'a u32>;
type RefOptRef<'a, T> = &'a Option<&'a T>;

fn foo(data: &Option<&u32>) {}

fn bar(data: &u32) -> &Option<&u32> {
    &None
}

struct StructRef<'a> {
    data: &'a Option<&'a u32>,
}

struct StructTupleRef<'a>(u32, &'a Option<&'a u32>);

enum EnumRef<'a> {
    Variant1(u32),
    Variant2(&'a Option<&'a u32>),
}

trait RefOptTrait {
    type A;
    fn foo(&self, _: Self::A);
}

impl RefOptTrait for u32 {
    type A = &'static Option<&'static Self>;

    fn foo(&self, _: Self::A) {}
}

fn main() {
    let x: &Option<&u32> = &None;
}
