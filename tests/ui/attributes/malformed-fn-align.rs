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

#[repr(align(16))] //~ ERROR `#[repr(align(...))]` is not supported on function items
fn f4() {}

#[rustc_align(16)] //~ ERROR `#[rustc_align(...)]` is not supported on struct items
struct S1;
