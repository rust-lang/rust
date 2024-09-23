#![feature(autodiff)]
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:autodiff_illegal.pp

// Test that invalid ad macros give nice errors and don't ICE.

// We can't use Duplicated on scalars
#[autodiff(df1, Reverse, Duplicated)]
pub fn f1(x: f64) {
    unimplemented!()
}

// We can't use Dual on scalars
#[autodiff(df2, Forward, Dual)]
pub fn f2(x: f64) {
    unimplemented!()
}

// Too many activities
#[autodiff(df3, Reverse, Duplicated, Const)]
pub fn f3(x: f64) {
    unimplemented!()
}

// To few activities
#[autodiff(df4, Reverse)]
pub fn f4(x: f64) {
    unimplemented!()
}

// We can't use Dual in Reverse mode
#[autodiff(df5, Reverse, Dual)]
pub fn f5(x: f64) {
    unimplemented!()
}

// We can't use Duplicated in Forward mode
#[autodiff(df6, Forward, Duplicated)]
pub fn f6(x: f64) {
    unimplemented!()
}

fn dummy() {
 
    #[autodiff(df7, Forward, Dual)]
    let mut x = 5;

    #[autodiff(df7, Forward, Dual)]
    x = x + 3;

    #[autodiff(df7, Forward, Dual)]
    let add_one_v2 = |x: u32| -> u32 { x + 1 };
}

// Malformed, where args?
#[autodiff]
pub fn f7(x: f64) {
    unimplemented!()
}

// Malformed, where args?
#[autodiff()]
pub fn f8(x: f64) {
    unimplemented!()
}

// Invalid attribute syntax
#[autodiff = ""]
pub fn f9(x: f64) {
    unimplemented!()
}

fn fn_exists() {}

// We colide with an already existing function
#[autodiff(fn_exists, Reverse, Active)]
pub fn f10(x: f64) {
    unimplemented!()
}

// Malformed, missing a mode
#[autodiff(df11)]
pub fn f11() {
    unimplemented!()
}

// Invalid Mode
#[autodiff(df12, Debug)]
pub fn f12() {
    unimplemented!()
}

// Invalid, please pick one Mode
// or use two autodiff macros.
#[autodiff(df13, Forward, Reverse)]
pub fn f13() {
    unimplemented!()
}

struct Foo {}

// We can't handle Active structs, because that would mean (in the general case), that we would
// need to allocate and initialize arbitrary user types. We have Duplicated/Dual input args for
// that. FIXME: Give a nicer error and suggest to the user to have a `&mut Foo` input instead.
#[autodiff(df14, Reverse, Active, Active)]
fn f14(x: f32) -> Foo {
    unimplemented!()
}

type MyFloat = f32;

// We would like to support this case in the future,
// but that requires us to implement our checks at a later stage
// like THIR which has type information available.
#[autodiff(df15, Reverse, Active, Active)]
fn f15(x: MyFloat) -> f32 {
    unimplemented!()
}

// We would like to support this case in the future
#[autodiff(df16, Reverse, Active, Active)]
fn f16(x: f32) -> MyFloat {
    unimplemented!()
}

#[repr(transparent)]
struct F64Trans { inner: f64 }

// We would like to support this case in the future
#[autodiff(df17, Reverse, Active, Active)]
fn f17(x: f64) -> F64Trans {
    unimplemented!()
}

// We would like to support this case in the future
#[autodiff(df18, Reverse, Active, Active)]
fn f18(x: F64Trans) -> f64 {
    unimplemented!()
}


fn main() {}
