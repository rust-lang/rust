#![feature(rustc_attrs)]
#![feature(optimize_attribute)]
#![feature(coverage_attribute)]
#![allow(unused_attributes)]

#[inline(always = 5)]
//~^ ERROR malformed
#[inline(always(x, y, z))]
//~^ ERROR malformed
#[instruction_set(arm::a32 = 5)]
//~^ ERROR malformed
#[instruction_set(arm::a32(x, y, z))]
//~^ ERROR malformed
#[optimize(size = 5)]
//~^ ERROR malformed
#[optimize(size(x, y, z))]
//~^ ERROR malformed
#[coverage(off = 5)]
//~^ ERROR malformed
#[coverage(off(x, y, z))]
//~^ ERROR malformed
fn main() {

}

#[macro_export(local_inner_macros = 5)]
//~^ ERROR valid forms for the attribute are
//~| WARN previously accepted
#[macro_export(local_inner_macros(x, y, z))]
//~^ ERROR valid forms for the attribute are
//~| WARN previously accepted
macro_rules! m {
    () => {};
}

#[rustc_allow_const_fn_unstable(x = 5)]
//~^ ERROR `rustc_allow_const_fn_unstable` expects feature names
#[rustc_allow_const_fn_unstable(x(x, y, z))]
//~^ ERROR `rustc_allow_const_fn_unstable` expects feature names
const fn g() {}

#[used(always = 5)]
//~^ ERROR malformed
#[used(always(x, y, z))]
//~^ ERROR malformed
static H: u64 = 5;

#[rustc_must_implement_one_of(eq = 5, neq)]
//~^ ERROR malformed
#[rustc_must_implement_one_of(eq(x, y, z), neq)]
//~^ ERROR malformed
trait T {

}

#[rustc_dump_layout(debug = 5)]
//~^ ERROR malformed
#[rustc_dump_layout(debug(x, y, z))]
//~^ ERROR malformed
enum E {

}