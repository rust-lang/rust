#![feature(fn_align)]
#![crate_type = "lib"]

trait MyTrait {
    #[align] //~ ERROR malformed `align` attribute input
    fn myfun();
}

#[align = 16] //~ ERROR malformed `align` attribute input
fn f1() {}

#[align("hello")] //~ ERROR invalid alignment value: not an unsuffixed integer
fn f2() {}

#[align(0)] //~ ERROR invalid alignment value: not a power of two
fn f3() {}

#[repr(align(16))] //~ ERROR `#[repr(align(...))]` is not supported on function items
fn f4() {}

#[align(16)] //~ ERROR `#[align(...)]` is not supported on struct items
struct S1;
