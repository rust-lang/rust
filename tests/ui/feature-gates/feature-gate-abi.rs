// gate-test-intrinsics
//@ compile-flags: --crate-type=rlib

#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized { }

#[lang="tuple_trait"]
trait Tuple { }

// Functions
extern "rust-intrinsic" fn f1() {} //~ ERROR intrinsics are subject to change
                                   //~^ ERROR intrinsic must be in
                                   //~| ERROR unrecognized intrinsic function: `f1`
extern "rust-intrinsic" fn f2() {} //~ ERROR intrinsics are subject to change
                                       //~^ ERROR intrinsic must be in
                                       //~| ERROR unrecognized intrinsic function: `f2`
extern "rust-call" fn f4(_: ()) {} //~ ERROR rust-call ABI is subject to change

// Methods in trait definition
trait Tr {
    extern "rust-intrinsic" fn m1(); //~ ERROR intrinsics are subject to change
                                     //~^ ERROR intrinsic must be in
    extern "rust-intrinsic" fn m2(); //~ ERROR intrinsics are subject to change
                                         //~^ ERROR intrinsic must be in
    extern "rust-call" fn m4(_: ()); //~ ERROR rust-call ABI is subject to change

    extern "rust-call" fn dm4(_: ()) {} //~ ERROR rust-call ABI is subject to change
}

struct S;

// Methods in trait impl
impl Tr for S {
    extern "rust-intrinsic" fn m1() {} //~ ERROR intrinsics are subject to change
                                       //~^ ERROR intrinsic must be in
    extern "rust-intrinsic" fn m2() {} //~ ERROR intrinsics are subject to change
                                           //~^ ERROR intrinsic must be in
    extern "rust-call" fn m4(_: ()) {} //~ ERROR rust-call ABI is subject to change
}

// Methods in inherent impl
impl S {
    extern "rust-intrinsic" fn im1() {} //~ ERROR intrinsics are subject to change
                                        //~^ ERROR intrinsic must be in
    extern "rust-intrinsic" fn im2() {} //~ ERROR intrinsics are subject to change
                                            //~^ ERROR intrinsic must be in
    extern "rust-call" fn im4(_: ()) {} //~ ERROR rust-call ABI is subject to change
}

// Function pointer types
type A1 = extern "rust-intrinsic" fn(); //~ ERROR intrinsics are subject to change
type A2 = extern "rust-intrinsic" fn(); //~ ERROR intrinsics are subject to change
type A4 = extern "rust-call" fn(_: ()); //~ ERROR rust-call ABI is subject to change

// Foreign modules
extern "rust-intrinsic" {} //~ ERROR intrinsics are subject to change
extern "rust-intrinsic" {} //~ ERROR intrinsics are subject to change
extern "rust-call" {} //~ ERROR rust-call ABI is subject to change
