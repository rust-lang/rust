#![deny(improper_ctypes, improper_c_fn_definitions, improper_ctype_definitions)]
#![deny(improper_c_callbacks)]

//@ aux-build: outer_crate_types.rs
//@ compile-flags:--extern outer_crate_types
extern crate outer_crate_types as outer;

// ////////////////////////////////////////////////////////
// first, the same bank of types as in the extern crate

#[repr(C)]
struct SafeStruct (i32);

#[repr(C)]
struct UnsafeStruct (String);
//~^ ERROR: `repr(C)` type uses type `String`

#[repr(C)]
#[allow(improper_ctype_definitions)]
struct AllowedUnsafeStruct (String);

// refs are only unsafe if the value comes from the other side of the FFI boundary
// due to the non-null assumption
// (technically there are also assumptions about non-dandling, alignment,
//  aliasing, lifetimes, etc...)
// the lint is not raised here, but will be if used in the wrong place
#[repr(C)]
struct UnsafeFromForeignStruct<'a> (&'a u32);

#[repr(C)]
#[allow(improper_ctype_definitions)]
struct AllowedUnsafeFromForeignStruct<'a> (&'a u32);


type SafeFnPtr = extern "C" fn(i32)->i32;

type UnsafeFnPtr = extern "C" fn((i32, i32))->i32;
//~^ ERROR: `extern` callback uses type `(i32, i32)`


// for now, let's not lint on the nonzero assumption,
// because:
// - we don't know if the callback is rust-callee-foreign-caller or the other way around
// - having to cast around function signatures to get function pointers
//   would be an awful experience
// so, let's assume that the unsafety in this fnptr
// will be pointed out indirectly by a lint elsewhere
// (note: there's one case where the error would be missed altogether:
//  a rust-caller,non-rust-callee callback where the fnptr
//  is given as an argument to a rust-callee,non-rust-caller
//  FFI boundary)
#[allow(improper_c_callbacks)]
type AllowedUnsafeFnPtr = extern "C" fn(&[i32])->i32;

type UnsafeRustCalleeFnPtr = extern "C" fn(i32)->&'static i32;

#[allow(improper_c_callbacks)]
type AllowedUnsafeRustCalleeFnPtr = extern "C" fn(i32)->&'static i32;

type UnsafeForeignCalleeFnPtr = extern "C" fn(&i32);

#[allow(improper_c_callbacks)]
type AllowedUnsafeForeignCalleeFnPtr = extern "C" fn(&i32);


// ////////////////////////////////////////////////////////
// then, some functions that use them

static INT: u32 = 42;

#[allow(improper_c_fn_definitions)]
extern "C" fn fn1a(e: &String) -> &str {&*e}
extern "C" fn fn1u(e: &String) -> &str {&*e}
//~^ ERROR: `extern` fn uses type `&str`
//~^^ ERROR: `extern` fn uses type `&String`

#[allow(improper_c_fn_definitions)]
extern "C" fn fn2a(e: UnsafeStruct) {}
extern "C" fn fn2u(e: UnsafeStruct) {}
//~^ ERROR: `extern` fn uses type `UnsafeStruct`
#[allow(improper_c_fn_definitions)]
extern "C" fn fn2oa(e: outer::UnsafeStruct) {}
extern "C" fn fn2ou(e: outer::UnsafeStruct) {}
//~^ ERROR: `extern` fn uses type `outer::UnsafeStruct`

#[allow(improper_c_fn_definitions)]
extern "C" fn fn3a(e: AllowedUnsafeStruct) {}
extern "C" fn fn3u(e: AllowedUnsafeStruct) {}
//~^ ERROR: `extern` fn uses type `AllowedUnsafeStruct`
// ^^ FIXME: ...ideally the lint should not trigger here
#[allow(improper_c_fn_definitions)]
extern "C" fn fn3oa(e: outer::AllowedUnsafeStruct) {}
extern "C" fn fn3ou(e: outer::AllowedUnsafeStruct) {}
//~^ ERROR: `extern` fn uses type `outer::AllowedUnsafeStruct`
// ^^ FIXME: ...ideally the lint should not trigger here

#[allow(improper_c_fn_definitions)]
extern "C" fn fn4a(e: UnsafeFromForeignStruct) {}
extern "C" fn fn4u(e: UnsafeFromForeignStruct) {}
//~^ ERROR: `extern` fn uses type `UnsafeFromForeignStruct<'_>`
#[allow(improper_c_fn_definitions)]
extern "C" fn fn4oa(e: outer::UnsafeFromForeignStruct) {}
extern "C" fn fn4ou(e: outer::UnsafeFromForeignStruct) {}
//~^ ERROR: `extern` fn uses type `outer::UnsafeFromForeignStruct<'_>`

#[allow(improper_c_fn_definitions)]
extern "C" fn fn5a() -> UnsafeFromForeignStruct<'static> { UnsafeFromForeignStruct(&INT)}
extern "C" fn fn5u() -> UnsafeFromForeignStruct<'static> { UnsafeFromForeignStruct(&INT)}
#[allow(improper_c_fn_definitions)]
extern "C" fn fn5oa() -> outer::UnsafeFromForeignStruct<'static> {
    outer::UnsafeFromForeignStruct(&INT)
}
extern "C" fn fn5ou() -> outer::UnsafeFromForeignStruct<'static> {
    outer::UnsafeFromForeignStruct(&INT)
}

#[allow(improper_c_fn_definitions)]
extern "C" fn fn6a() -> AllowedUnsafeFromForeignStruct<'static> {
    AllowedUnsafeFromForeignStruct(&INT)
}
extern "C" fn fn6u() -> AllowedUnsafeFromForeignStruct<'static> {
    AllowedUnsafeFromForeignStruct(&INT)
}
#[allow(improper_c_fn_definitions)]
extern "C" fn fn6oa() -> outer::AllowedUnsafeFromForeignStruct<'static> {
    outer::AllowedUnsafeFromForeignStruct(&INT)
}
extern "C" fn fn6ou() -> outer::AllowedUnsafeFromForeignStruct<'static> {
    outer::AllowedUnsafeFromForeignStruct(&INT)
}

// ////////////////////////////////////////////////////////
// special cases: struct-in-fnptr and fnptr-in-struct

#[repr(C)]
struct FakeVTable<A>{
//~^ ERROR: `repr(C)` type uses type `(A, usize)`
    make_new: extern "C" fn() -> A,
    combine: extern "C" fn(&[A]) -> A,
    //~^ ERROR: `extern` callback uses type `&[A]`
    drop: extern "C" fn(A),
    something_else: (A, usize),
}

type FakeVTableMaker = extern "C" fn() -> FakeVTable<u32>;
//~^ ERROR: `extern` callback uses type `FakeVTable<u32>`

#[repr(C)]
#[allow(improper_c_callbacks, improper_ctype_definitions)]
struct FakeVTableAllowed<A>{
    make_new: extern "C" fn() -> A,
    combine: extern "C" fn(&[A]) -> A,
    drop: extern "C" fn(A),
    something_else: (A, usize),
}

#[allow(improper_c_callbacks)]
type FakeVTableMakerAllowed = extern "C" fn() -> FakeVTable<u32>;

fn main(){}
