// ignore-tidy-linelength

// FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
#![feature(rustc_attrs)]
#![feature(fn_align)]

#![crate_type = "lib"]

trait MyTrait {
    #[rustc_align] //~ ERROR malformed `rustc_align` attribute input
    fn myfun1();

    #[rustc_align(1, 2)] //~ ERROR malformed `rustc_align` attribute input
    fn myfun2();
}

#[rustc_align = 16] //~ ERROR malformed `rustc_align` attribute input
fn f1() {}

#[rustc_align("hello")] //~ ERROR invalid alignment value: not an unsuffixed integer
fn f2() {}

#[rustc_align(0)] //~ ERROR invalid alignment value: not a power of two
fn f3() {}

#[repr(align(16))] //~ ERROR `#[repr(align(...))]` is not supported on functions
fn f4() {}

#[rustc_align(-1)] //~ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `-`
fn f5() {}

#[rustc_align(3)] //~ ERROR invalid alignment value: not a power of two
fn f6() {}

#[rustc_align(4usize)] //~ ERROR invalid alignment value: not an unsuffixed integer [E0589]
//~^ ERROR suffixed literals are not allowed in attributes
fn f7() {}

#[rustc_align(16)]
#[rustc_align(3)] //~ ERROR invalid alignment value: not a power of two
#[rustc_align(16)]
fn f8() {}

#[rustc_align(16)] //~ ERROR attribute cannot be used on
struct S1;

#[rustc_align(32)] //~ ERROR attribute cannot be used on
const FOO: i32 = 42;

#[rustc_align(32)] //~ ERROR attribute cannot be used on
mod test {}

#[rustc_align(32)] //~ ERROR attribute cannot be used on
use ::std::iter;
