// gate-test-intrinsics
//@ compile-flags: --crate-type=rlib

#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized { }

#[lang="tuple_trait"]
trait Tuple { }

// Functions
extern "rust-intrinsic" fn f1() {} //~ ERROR extern "rust-intrinsic" ABI is an implementation detail
                                   //~^ ERROR intrinsic must be in
extern "rust-intrinsic" fn f2() {} //~ ERROR extern "rust-intrinsic" ABI is an implementation detail
                                       //~^ ERROR intrinsic must be in
extern "rust-call" fn f4(_: ()) {} //~ ERROR extern "rust-call" ABI is experimental and subject to change

// Methods in trait definition
trait Tr {
    extern "rust-intrinsic" fn m1(); //~ ERROR extern "rust-intrinsic" ABI is an implementation detail
                                     //~^ ERROR intrinsic must be in
    extern "rust-intrinsic" fn m2(); //~ ERROR extern "rust-intrinsic" ABI is an implementation detail
                                         //~^ ERROR intrinsic must be in
    extern "rust-call" fn m4(_: ()); //~ ERROR extern "rust-call" ABI is experimental and subject to change

    extern "rust-call" fn dm4(_: ()) {} //~ ERROR extern "rust-call" ABI is experimental and subject to change
}

struct S;

// Methods in trait impl
impl Tr for S {
    extern "rust-intrinsic" fn m1() {} //~ ERROR extern "rust-intrinsic" ABI is an implementation detail
                                       //~^ ERROR intrinsic must be in
    extern "rust-intrinsic" fn m2() {} //~ ERROR extern "rust-intrinsic" ABI is an implementation detail
                                           //~^ ERROR intrinsic must be in
    extern "rust-call" fn m4(_: ()) {} //~ ERROR extern "rust-call" ABI is experimental and subject to change
}

// Methods in inherent impl
impl S {
    extern "rust-intrinsic" fn im1() {} //~ ERROR extern "rust-intrinsic" ABI is an implementation detail
                                        //~^ ERROR intrinsic must be in
    extern "rust-intrinsic" fn im2() {} //~ ERROR extern "rust-intrinsic" ABI is an implementation detail
                                            //~^ ERROR intrinsic must be in
    extern "rust-call" fn im4(_: ()) {} //~ ERROR extern "rust-call" ABI is experimental and subject to change
}

// Function pointer types
type A1 = extern "rust-intrinsic" fn(); //~ ERROR extern "rust-intrinsic" ABI is an implementation detail
type A2 = extern "rust-intrinsic" fn(); //~ ERROR extern "rust-intrinsic" ABI is an implementation detail
type A4 = extern "rust-call" fn(_: ()); //~ ERROR extern "rust-call" ABI is experimental and subject to change

// Foreign modules
extern "rust-intrinsic" {} //~ ERROR extern "rust-intrinsic" ABI is an implementation detail
extern "rust-intrinsic" {} //~ ERROR extern "rust-intrinsic" ABI is an implementation detail
extern "rust-call" {} //~ ERROR extern "rust-call" ABI is experimental and subject to change
