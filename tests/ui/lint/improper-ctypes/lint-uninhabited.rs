#![feature(never_type)]

#![allow(dead_code, unused_variables)]
#![deny(improper_ctypes)]
#![deny(improper_c_fn_definitions, improper_c_callbacks)]

use std::mem::transmute;

enum Uninhabited{}

#[repr(C)]
struct AlsoUninhabited{
    a: Uninhabited,
    b: i32,
}

#[repr(C)]
enum Inhabited{
    OhNo(Uninhabited),
    OhYes(i32),
}

struct EmptyRust;

#[repr(transparent)]
struct HalfHiddenUninhabited {
    is_this_a_tuple: (i8,i8),
    zst_inh: EmptyRust,
    zst_uninh: !,
}

extern "C" {

fn bad_entry(e: AlsoUninhabited); //~ ERROR: uses type `AlsoUninhabited`
fn bad_exit()->AlsoUninhabited;

fn bad0_entry(e: Uninhabited); //~ ERROR: uses type `Uninhabited`
fn bad0_exit()->Uninhabited;

fn good_entry(e: Inhabited);
fn good_exit()->Inhabited;

fn never_entry(e:!); //~ ERROR: uses type `!`
fn never_exit()->!;

}

extern "C" fn impl_bad_entry(e: AlsoUninhabited) {} //~ ERROR: uses type `AlsoUninhabited`
extern "C" fn impl_bad_exit()->AlsoUninhabited {
    AlsoUninhabited{
        a: impl_bad0_exit(),
        b: 0,
    }
}

extern "C" fn impl_bad0_entry(e: Uninhabited) {} //~ ERROR: uses type `Uninhabited`
extern "C" fn impl_bad0_exit()->Uninhabited {
    unsafe{transmute(())} //~ WARN: does not permit zero-initialization
}

extern "C" fn impl_good_entry(e: Inhabited) {}
extern "C" fn impl_good_exit() -> Inhabited {
    Inhabited::OhYes(0)
}

extern "C" fn impl_never_entry(e:!){} //~ ERROR: uses type `!`
extern "C" fn impl_never_exit()->! {
    loop{}
}

extern "C" fn weird_pattern(e:HalfHiddenUninhabited){}
//~^ ERROR: uses type `HalfHiddenUninhabited`


fn main(){}
