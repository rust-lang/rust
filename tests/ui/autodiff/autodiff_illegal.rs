//@ needs-enzyme

#![feature(autodiff)]
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:autodiff_illegal.pp

// Test that invalid ad macros give nice errors and don't ICE.

use std::autodiff::{autodiff_forward, autodiff_reverse};

// We can't use Duplicated on scalars
#[autodiff_reverse(df1, Duplicated)]
pub fn f1(x: f64) {
    //~^ ERROR     Duplicated can not be used for this type
    unimplemented!()
}

// Too many activities
#[autodiff_reverse(df3, Duplicated, Const)]
pub fn f3(x: f64) {
    //~^^ ERROR     expected 1 activities, but found 2
    unimplemented!()
}

// To few activities
#[autodiff_reverse(df4)]
pub fn f4(x: f64) {
    //~^^ ERROR     expected 1 activities, but found 0
    unimplemented!()
}

// We can't use Dual in Reverse mode
#[autodiff_reverse(df5, Dual)]
pub fn f5(x: f64) {
    //~^^ ERROR     Dual can not be used in Reverse Mode
    unimplemented!()
}

// We can't use Duplicated in Forward mode
#[autodiff_forward(df6, Duplicated)]
pub fn f6(x: f64) {
    //~^^ ERROR Duplicated can not be used in Forward Mode
    //~^^ ERROR Duplicated can not be used for this type
    unimplemented!()
}

fn dummy() {
    #[autodiff_forward(df7, Dual)]
    let mut x = 5;
    //~^ ERROR autodiff must be applied to function

    #[autodiff_forward(df7, Dual)]
    x = x + 3;
    //~^^ ERROR attributes on expressions are experimental [E0658]
    //~^^ ERROR autodiff must be applied to function

    #[autodiff_forward(df7, Dual)]
    let add_one_v2 = |x: u32| -> u32 { x + 1 };
    //~^ ERROR autodiff must be applied to function
}

// Malformed, where args?
#[autodiff_forward]
pub fn f7(x: f64) {
    //~^ ERROR autodiff requires at least a name and mode
    unimplemented!()
}

// Malformed, where args?
#[autodiff_forward()]
pub fn f8(x: f64) {
    //~^ ERROR autodiff requires at least a name and mode
    unimplemented!()
}

// Invalid attribute syntax
#[autodiff_forward = ""]
pub fn f9(x: f64) {
    //~^ ERROR autodiff requires at least a name and mode
    unimplemented!()
}

fn fn_exists() {}

// We colide with an already existing function
#[autodiff_reverse(fn_exists, Active)]
pub fn f10(x: f64) {
    //~^^ ERROR the name `fn_exists` is defined multiple times [E0428]
    unimplemented!()
}

// Invalid, please pick one Mode
// or use two autodiff macros.
#[autodiff_reverse(df13, Reverse)]
pub fn f13() {
    //~^^ ERROR did not recognize Activity: `Reverse`
    unimplemented!()
}

struct Foo {}

// We can't handle Active structs, because that would mean (in the general case), that we would
// need to allocate and initialize arbitrary user types. We have Duplicated/Dual input args for
// that. FIXME: Give a nicer error and suggest to the user to have a `&mut Foo` input instead.
#[autodiff_reverse(df14, Active, Active)]
fn f14(x: f32) -> Foo {
    unimplemented!()
}

type MyFloat = f32;

// We would like to support type alias to f32/f64 in argument type in the future,
// but that requires us to implement our checks at a later stage
// like THIR which has type information available.
#[autodiff_reverse(df15, Active, Active)]
fn f15(x: MyFloat) -> f32 {
    //~^^ ERROR failed to resolve: use of undeclared type `MyFloat` [E0433]
    unimplemented!()
}

// We would like to support type alias to f32/f64 in return type in the future
#[autodiff_reverse(df16, Active, Active)]
fn f16(x: f32) -> MyFloat {
    unimplemented!()
}

#[repr(transparent)]
struct F64Trans {
    inner: f64,
}

// We would like to support `#[repr(transparent)]` f32/f64 wrapper in return type in the future
#[autodiff_reverse(df17, Active, Active)]
fn f17(x: f64) -> F64Trans {
    unimplemented!()
}

// We would like to support `#[repr(transparent)]` f32/f64 wrapper in argument type in the future
#[autodiff_reverse(df18, Active, Active)]
fn f18(x: F64Trans) -> f64 {
    //~^^ ERROR failed to resolve: use of undeclared type `F64Trans` [E0433]
    unimplemented!()
}

// Invalid return activity
#[autodiff_forward(df19, Dual, Active)]
fn f19(x: f32) -> f32 {
    //~^^ ERROR invalid return activity Active in Forward Mode
    unimplemented!()
}

#[autodiff_reverse(df20, Active, Dual)]
fn f20(x: f32) -> f32 {
    //~^^ ERROR invalid return activity Dual in Reverse Mode
    unimplemented!()
}

// Duplicated cannot be used as return activity
#[autodiff_reverse(df21, Active, Duplicated)]
fn f21(x: f32) -> f32 {
    //~^^ ERROR invalid return activity Duplicated in Reverse Mode
    unimplemented!()
}

struct DoesNotImplDefault;
#[autodiff_forward(df22, Dual)]
pub fn f22() -> DoesNotImplDefault {
    //~^^ ERROR the function or associated item `default` exists for tuple `(DoesNotImplDefault, DoesNotImplDefault)`, but its trait bounds were not satisfied
    unimplemented!()
}

fn main() {}
