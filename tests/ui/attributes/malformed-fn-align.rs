#![feature(fn_align)]
#![crate_type = "lib"]

trait MyTrait {
    #[align] //~ ERROR malformed `align` attribute input
    fn myfun1();

    #[align(1, 2)] //~ ERROR malformed `align` attribute input
    fn myfun2();
}

#[align = 16] //~ ERROR malformed `align` attribute input
fn f1() {}

#[align("hello")] //~ ERROR invalid alignment value: not an unsuffixed integer
fn f2() {}

#[align(0)] //~ ERROR invalid alignment value: not a power of two
fn f3() {}

#[repr(align(16))] //~ ERROR `#[repr(align(...))]` is not supported on function items
fn f4() {}

#[align(-1)] //~ ERROR expected unsuffixed literal, found `-`
fn f5() {}

#[align(3)] //~ ERROR invalid alignment value: not a power of two
fn f6() {}

#[align(4usize)] //~ ERROR invalid alignment value: not an unsuffixed integer [E0589]
//~^ ERROR suffixed literals are not allowed in attributes
fn f7() {}

#[align(16)]
#[align(3)] //~ ERROR invalid alignment value: not a power of two
#[align(16)]
fn f8() {}

#[align(16)] //~ ERROR `#[align(...)]` is not supported on struct items
struct S1;

#[align(32)] //~ ERROR `#[align(...)]` should be applied to a function item
const FOO: i32 = 42;

#[align(32)] //~ ERROR `#[align(...)]` should be applied to a function item
mod test {}

#[align(32)] //~ ERROR `#[align(...)]` should be applied to a function item
use ::std::iter;
