#![allow(unused_unsafe)]
#![allow(dead_code)]
#![deny(unsafe_code)]
#![feature(naked_functions_rustic_abi)]
#![feature(ffi_pure)]
#![feature(ffi_const)]

use std::arch::naked_asm;

struct Bar;
struct Bar2;
struct Bar3;

#[allow(unsafe_code)]
mod allowed_unsafe {
    fn allowed() { unsafe {} }
    unsafe fn also_allowed() {}
    unsafe trait AllowedUnsafe { }
    unsafe impl AllowedUnsafe for super::Bar {}
    #[no_mangle] fn allowed2() {}
    #[export_name = "foo"] fn allowed3() {}
}

macro_rules! unsafe_in_macro {
    () => {{
        #[no_mangle] fn foo() {} //~ ERROR: usage of the unsafe `#[no_mangle]` attribute
        #[no_mangle] static FOO: u32 = 5; //~ ERROR: usage of the unsafe `#[no_mangle]` attribute
        #[export_name = "bar"] fn bar() {}
        //~^ ERROR: usage of the unsafe `#[export_name]` attribute
        #[export_name = "BAR"] static BAR: u32 = 5;
        //~^ ERROR: usage of the unsafe `#[export_name]` attribute
        unsafe {} //~ ERROR: usage of an `unsafe` block
    }}
}

#[no_mangle] fn foo() {} //~ ERROR: usage of the unsafe `#[no_mangle]` attribute
#[no_mangle] static FOO: u32 = 5; //~ ERROR: usage of the unsafe `#[no_mangle]` attribute

trait AssocFnTrait {
    fn foo();
}

struct AssocFnFoo;

impl AssocFnFoo {
    #[no_mangle] fn foo() {} //~ ERROR: usage of the unsafe `#[no_mangle]` attribute
}

impl AssocFnTrait for AssocFnFoo {
    #[no_mangle] fn foo() {} //~ ERROR: usage of the unsafe `#[no_mangle]` attribute
}

#[export_name = "bar"] fn bar() {} //~ ERROR: usage of the unsafe `#[export_name]` attribute
#[export_name = "BAR"] static BAR: u32 = 5; //~ ERROR: usage of the unsafe `#[export_name]` attribute

#[link_section = "__TEXT,__text"] fn uwu() {} //~ ERROR: usage of the unsafe `#[link_section]` attribute
#[link_section = "__TEXT,__text"] static UWU: u32 = 5; //~ ERROR: usage of the unsafe `#[link_section]` attribute

struct AssocFnBar;

impl AssocFnBar {
    #[export_name = "bar"] fn bar() {} //~ ERROR: usage of the unsafe `#[export_name]` attribute
}

impl AssocFnTrait for AssocFnBar {
    #[export_name = "bar"] fn foo() {} //~ ERROR: usage of the unsafe `#[export_name]` attribute
}

unsafe fn baz() {} //~ ERROR: declaration of an `unsafe` function
unsafe trait Foo {} //~ ERROR: declaration of an `unsafe` trait
unsafe impl Foo for Bar {} //~ ERROR: implementation of an `unsafe` trait

trait Baz {
    unsafe fn baz(&self); //~ ERROR: declaration of an `unsafe` method
    unsafe fn provided(&self) {} //~ ERROR: implementation of an `unsafe` method
    unsafe fn provided_override(&self) {} //~ ERROR: implementation of an `unsafe` method
}

impl Baz for Bar {
    unsafe fn baz(&self) {} //~ ERROR: implementation of an `unsafe` method
    unsafe fn provided_override(&self) {} //~ ERROR: implementation of an `unsafe` method
}


#[allow(unsafe_code)]
trait A {
    unsafe fn allowed_unsafe(&self);
    unsafe fn allowed_unsafe_provided(&self) {}
}

#[allow(unsafe_code)]
impl Baz for Bar2 {
    unsafe fn baz(&self) {}
    unsafe fn provided_override(&self) {}
}

impl Baz for Bar3 {
    #[allow(unsafe_code)]
    unsafe fn baz(&self) {}
    unsafe fn provided_override(&self) {} //~ ERROR: implementation of an `unsafe` method
}

#[allow(unsafe_code)]
unsafe trait B {
    fn dummy(&self) {}
}

trait C {
    #[allow(unsafe_code)]
    unsafe fn baz(&self);
    unsafe fn provided(&self) {} //~ ERROR: implementation of an `unsafe` method
}

impl C for Bar {
    #[allow(unsafe_code)]
    unsafe fn baz(&self) {}
    unsafe fn provided(&self) {} //~ ERROR: implementation of an `unsafe` method
}

impl C for Bar2 {
    unsafe fn baz(&self) {} //~ ERROR: implementation of an `unsafe` method
}

trait D {
    #[allow(unsafe_code)]
    unsafe fn unsafe_provided(&self) {}
}

impl D for Bar {}

fn main() {
    unsafe {} //~ ERROR: usage of an `unsafe` block

    unsafe_in_macro!()
}

#[unsafe(naked)] fn naked1() { naked_asm!("halt") }
//~^ ERROR usage of the unsafe `#[naked]` attribute

struct Naked;
impl Naked {
    #[unsafe(naked)] fn naked2() { naked_asm!("halt") }
    //~^ ERROR usage of the unsafe `#[naked]` attribute
}

trait NakedTrait {
    #[unsafe(naked)] fn naked3() { naked_asm!("halt") }
    //~^ ERROR usage of the unsafe `#[naked]` attribute
    fn naked4();
}
impl NakedTrait for Naked {
    #[unsafe(naked)] fn naked4() { naked_asm!("halt") }
    //~^ ERROR usage of the unsafe `#[naked]` attribute
}

extern "C" {
    #[unsafe(ffi_pure)]
    //~^ ERROR usage of the unsafe `#[ffi_pure]` attribute
    fn ffi_pure();

    #[unsafe(ffi_const)]
    //~^ ERROR usage of the unsafe `#[ffi_const]` attribute
    fn ffi_const();
}
