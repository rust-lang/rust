#![feature(rustc_attrs)]

#[inline(always = 5)]
//~^ ERROR malformed
#[inline(always(x, y, z))]
//~^ ERROR malformed
#[instruction_set(arm::a32 = 5)]
//~^ ERROR malformed
#[instruction_set(arm::a32(x, y, z))]
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