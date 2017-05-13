// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// ignore-lldb
// ignore-android: FIXME(#24958)
// ignore-arm: FIXME(#24958)
// ignore-aarch64: FIXME(#24958)

// compile-flags:-g

// gdb-command:run

// STRUCTS
// gdb-command:whatis simple_struct
// gdbg-check:type = struct Struct1
// gdbr-check:type = type_names::Struct1

// gdb-command:whatis generic_struct1
// gdbg-check:type = struct GenericStruct<type_names::mod1::Struct2, type_names::mod1::mod2::Struct3>
// gdbr-check:type = type_names::GenericStruct<type_names::mod1::Struct2, type_names::mod1::mod2::Struct3>

// gdb-command:whatis generic_struct2
// gdbg-check:type = struct GenericStruct<type_names::Struct1, extern "fastcall" fn(isize) -> usize>
// gdbr-check:type = type_names::GenericStruct<type_names::Struct1, extern "fastcall" fn(isize) -> usize>

// gdb-command:whatis mod_struct
// gdbg-check:type = struct Struct2
// gdbr-check:type = type_names::mod1::Struct2


// ENUMS
// gdb-command:whatis simple_enum_1
// gdbg-check:type = union Enum1
// gdbr-check:type = type_names::Enum1

// gdb-command:whatis simple_enum_2
// gdbg-check:type = union Enum1
// gdbr-check:type = type_names::Enum1

// gdb-command:whatis simple_enum_3
// gdbg-check:type = union Enum2
// gdbr-check:type = type_names::mod1::Enum2

// gdb-command:whatis generic_enum_1
// gdbg-check:type = union Enum3<type_names::mod1::Struct2>
// gdbr-check:type = type_names::mod1::mod2::Enum3<type_names::mod1::Struct2>

// gdb-command:whatis generic_enum_2
// gdbg-check:type = union Enum3<type_names::Struct1>
// gdbr-check:type = type_names::mod1::mod2::Enum3<type_names::Struct1>


// TUPLES
// gdb-command:whatis tuple1
// gdbg-check:type = struct (u32, type_names::Struct1, type_names::mod1::mod2::Enum3<type_names::mod1::Struct2>)
// gdbr-check:type = (u32, type_names::Struct1, type_names::mod1::mod2::Enum3<type_names::mod1::Struct2>)

// gdb-command:whatis tuple2
// gdbg-check:type = struct ((type_names::Struct1, type_names::mod1::mod2::Struct3), type_names::mod1::Enum2, char)
// gdbr-check:type = ((type_names::Struct1, type_names::mod1::mod2::Struct3), type_names::mod1::Enum2, char)


// BOX
// gdb-command:whatis box1
// gdbg-check:type = struct (Box<f32>, i32)
// gdbr-check:type = (Box<f32>, i32)

// gdb-command:whatis box2
// gdbg-check:type = struct (Box<type_names::mod1::mod2::Enum3<f32>>, i32)
// gdbr-check:type = (Box<type_names::mod1::mod2::Enum3<f32>>, i32)


// REFERENCES
// gdb-command:whatis ref1
// gdbg-check:type = struct (&type_names::Struct1, i32)
// gdbr-check:type = (&type_names::Struct1, i32)

// gdb-command:whatis ref2
// gdbg-check:type = struct (&type_names::GenericStruct<char, type_names::Struct1>, i32)
// gdbr-check:type = (&type_names::GenericStruct<char, type_names::Struct1>, i32)

// gdb-command:whatis mut_ref1
// gdbg-check:type = struct (&mut type_names::Struct1, i32)
// gdbr-check:type = (&mut type_names::Struct1, i32)

// gdb-command:whatis mut_ref2
// gdbg-check:type = struct (&mut type_names::GenericStruct<type_names::mod1::Enum2, f64>, i32)
// gdbr-check:type = (&mut type_names::GenericStruct<type_names::mod1::Enum2, f64>, i32)


// RAW POINTERS
// gdb-command:whatis mut_ptr1
// gdbg-check:type = struct (*mut type_names::Struct1, isize)
// gdbr-check:type = (*mut type_names::Struct1, isize)

// gdb-command:whatis mut_ptr2
// gdbg-check:type = struct (*mut isize, isize)
// gdbr-check:type = (*mut isize, isize)

// gdb-command:whatis mut_ptr3
// gdbg-check:type = struct (*mut type_names::mod1::mod2::Enum3<type_names::Struct1>, isize)
// gdbr-check:type = (*mut type_names::mod1::mod2::Enum3<type_names::Struct1>, isize)

// gdb-command:whatis const_ptr1
// gdbg-check:type = struct (*const type_names::Struct1, isize)
// gdbr-check:type = (*const type_names::Struct1, isize)

// gdb-command:whatis const_ptr2
// gdbg-check:type = struct (*const isize, isize)
// gdbr-check:type = (*const isize, isize)

// gdb-command:whatis const_ptr3
// gdbg-check:type = struct (*const type_names::mod1::mod2::Enum3<type_names::Struct1>, isize)
// gdbr-check:type = (*const type_names::mod1::mod2::Enum3<type_names::Struct1>, isize)


// VECTORS
// gdb-command:whatis fixed_size_vec1
// gdbg-check:type = struct ([type_names::Struct1; 3], i16)
// gdbr-check:type = ([type_names::Struct1; 3], i16)

// gdb-command:whatis fixed_size_vec2
// gdbg-check:type = struct ([usize; 3], i16)
// gdbr-check:type = ([usize; 3], i16)

// gdb-command:whatis slice1
// gdbg-check:type = struct &[usize]
// gdbr-check:type = &[usize]

// gdb-command:whatis slice2
// gdbg-check:type = struct &[type_names::mod1::Enum2]
// gdbr-check:type = &[type_names::mod1::Enum2]


// TRAITS
// gdb-command:whatis box_trait
// gdbg-check:type = struct Box<Trait1>
// gdbr-check:type = type_names::Box<Trait1>

// gdb-command:whatis ref_trait
// gdbg-check:type = struct &Trait1
// gdbr-check:type = type_names::&Trait1

// gdb-command:whatis mut_ref_trait
// gdbg-check:type = struct &mut Trait1
// gdbr-check:type = type_names::&mut Trait1

// gdb-command:whatis generic_box_trait
// gdbg-check:type = struct Box<Trait2<i32, type_names::mod1::Struct2>>
// gdbr-check:type = type_names::Box<Trait2<i32, type_names::mod1::Struct2>>

// gdb-command:whatis generic_ref_trait
// gdbg-check:type = struct &Trait2<type_names::Struct1, type_names::Struct1>
// gdbr-check:type = type_names::&Trait2<type_names::Struct1, type_names::Struct1>

// gdb-command:whatis generic_mut_ref_trait
// gdbg-check:type = struct &mut Trait2<type_names::mod1::mod2::Struct3, type_names::GenericStruct<usize, isize>>
// gdbr-check:type = type_names::&mut Trait2<type_names::mod1::mod2::Struct3, type_names::GenericStruct<usize, isize>>


// BARE FUNCTIONS
// gdb-command:whatis rust_fn
// gdbg-check:type = struct (fn(core::option::Option<isize>, core::option::Option<&type_names::mod1::Struct2>), usize)
// gdbr-check:type = (fn(core::option::Option<isize>, core::option::Option<&type_names::mod1::Struct2>), usize)

// gdb-command:whatis extern_c_fn
// gdbg-check:type = struct (extern "C" fn(isize), usize)
// gdbr-check:type = (extern "C" fn(isize), usize)

// gdb-command:whatis unsafe_fn
// gdbg-check:type = struct (unsafe fn(core::result::Result<char, f64>), usize)
// gdbr-check:type = (unsafe fn(core::result::Result<char, f64>), usize)

// gdb-command:whatis extern_stdcall_fn
// gdbg-check:type = struct (extern "stdcall" fn(), usize)
// gdbr-check:type = (extern "stdcall" fn(), usize)

// gdb-command:whatis rust_fn_with_return_value
// gdbg-check:type = struct (fn(f64) -> usize, usize)
// gdbr-check:type = (fn(f64) -> usize, usize)

// gdb-command:whatis extern_c_fn_with_return_value
// gdbg-check:type = struct (extern "C" fn() -> type_names::Struct1, usize)
// gdbr-check:type = (extern "C" fn() -> type_names::Struct1, usize)

// gdb-command:whatis unsafe_fn_with_return_value
// gdbg-check:type = struct (unsafe fn(type_names::GenericStruct<u16, u8>) -> type_names::mod1::Struct2, usize)
// gdbr-check:type = (unsafe fn(type_names::GenericStruct<u16, u8>) -> type_names::mod1::Struct2, usize)

// gdb-command:whatis extern_stdcall_fn_with_return_value
// gdbg-check:type = struct (extern "stdcall" fn(Box<isize>) -> usize, usize)
// gdbr-check:type = (extern "stdcall" fn(Box<isize>) -> usize, usize)

// gdb-command:whatis generic_function_int
// gdbg-check:type = struct (fn(isize) -> isize, usize)
// gdbr-check:type = (fn(isize) -> isize, usize)

// gdb-command:whatis generic_function_struct3
// gdbg-check:type = struct (fn(type_names::mod1::mod2::Struct3) -> type_names::mod1::mod2::Struct3, usize)
// gdbr-check:type = (fn(type_names::mod1::mod2::Struct3) -> type_names::mod1::mod2::Struct3, usize)

// gdb-command:whatis variadic_function
// gdbg-check:type = struct (unsafe extern "C" fn(*const u8, ...) -> isize, usize)
// gdbr-check:type = (unsafe extern "C" fn(*const u8, ...) -> isize, usize)


// CLOSURES
// gdb-command:whatis closure1
// gdbg-check:type = struct (closure, usize)
// gdbr-check:type = (closure, usize)

// gdb-command:whatis closure2
// gdbg-check:type = struct (closure, usize)
// gdbr-check:type = (closure, usize)

#![feature(box_syntax)]
#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

use self::Enum1::{Variant1, Variant2};
use std::marker::PhantomData;
use std::ptr;

pub struct Struct1;
struct GenericStruct<T1, T2>(PhantomData<(T1,T2)>);

enum Enum1 {
    Variant1,
    Variant2(isize),
}

mod mod1 {
    pub use self::Enum2::{Variant1, Variant2};
    pub struct Struct2;

    pub enum Enum2 {
        Variant1,
        Variant2(super::Struct1),
    }

    pub mod mod2 {
        pub use self::Enum3::{Variant1, Variant2};
        pub struct Struct3;

        pub enum Enum3<T> {
            Variant1,
            Variant2(T),
        }
    }
}

trait Trait1 { fn dummy(&self) { } }
trait Trait2<T1, T2> { fn dummy(&self, _: T1, _:T2) { } }

impl Trait1 for isize {}
impl<T1, T2> Trait2<T1, T2> for isize {}

fn rust_fn(_: Option<isize>, _: Option<&mod1::Struct2>) {}
extern "C" fn extern_c_fn(_: isize) {}
unsafe fn unsafe_fn(_: Result<char, f64>) {}
extern "stdcall" fn extern_stdcall_fn() {}

fn rust_fn_with_return_value(_: f64) -> usize { 4 }
extern "C" fn extern_c_fn_with_return_value() -> Struct1 { Struct1 }
unsafe fn unsafe_fn_with_return_value(_: GenericStruct<u16, u8>) -> mod1::Struct2 { mod1::Struct2 }
extern "stdcall" fn extern_stdcall_fn_with_return_value(_: Box<isize>) -> usize { 0 }

fn generic_function<T>(x: T) -> T { x }

#[allow(improper_ctypes)]
extern {
    fn printf(_:*const u8, ...) -> isize;
}

// In many of the cases below, the type that is actually under test is wrapped
// in a tuple, e.g. Box<T>, references, raw pointers, fixed-size vectors, ...
// This is because GDB will not print the type name from DWARF debuginfo for
// some kinds of types (pointers, arrays, functions, ...)
// Since tuples are structs as far as GDB is concerned, their name will be
// printed correctly, so the tests below just construct a tuple type that will
// then *contain* the type name that we want to see.
fn main() {

    // Structs
    let simple_struct = Struct1;
    let generic_struct1: GenericStruct<mod1::Struct2, mod1::mod2::Struct3> =
        GenericStruct(PhantomData);
    let generic_struct2: GenericStruct<Struct1, extern "fastcall" fn(isize) -> usize> =
        GenericStruct(PhantomData);
    let mod_struct = mod1::Struct2;

    // Enums
    let simple_enum_1 = Variant1;
    let simple_enum_2 = Variant2(0);
    let simple_enum_3 = mod1::Variant2(Struct1);

    let generic_enum_1: mod1::mod2::Enum3<mod1::Struct2> = mod1::mod2::Variant1;
    let generic_enum_2 = mod1::mod2::Variant2(Struct1);

    // Tuples
    let tuple1 = (8u32, Struct1, mod1::mod2::Variant2(mod1::Struct2));
    let tuple2 = ((Struct1, mod1::mod2::Struct3), mod1::Variant1, 'x');

    // Box
    let box1 = (box 1f32, 0i32);
    let box2 = (box mod1::mod2::Variant2(1f32), 0i32);

    // References
    let ref1 = (&Struct1, 0i32);
    let ref2 = (&GenericStruct::<char, Struct1>(PhantomData), 0i32);

    let mut mut_struct1 = Struct1;
    let mut mut_generic_struct = GenericStruct::<mod1::Enum2, f64>(PhantomData);
    let mut_ref1 = (&mut mut_struct1, 0i32);
    let mut_ref2 = (&mut mut_generic_struct, 0i32);

    // Raw Pointers
    let mut_ptr1: (*mut Struct1, isize) = (ptr::null_mut(), 0);
    let mut_ptr2: (*mut isize, isize) = (ptr::null_mut(), 0);
    let mut_ptr3: (*mut mod1::mod2::Enum3<Struct1>, isize) = (ptr::null_mut(), 0);

    let const_ptr1: (*const Struct1, isize) = (ptr::null(), 0);
    let const_ptr2: (*const isize, isize) = (ptr::null(), 0);
    let const_ptr3: (*const mod1::mod2::Enum3<Struct1>, isize) = (ptr::null(), 0);

    // Vectors
    let fixed_size_vec1 = ([Struct1, Struct1, Struct1], 0i16);
    let fixed_size_vec2 = ([0_usize, 1, 2], 0i16);

    let vec1 = vec![0_usize, 2, 3];
    let slice1 = &*vec1;
    let vec2 = vec![mod1::Variant2(Struct1)];
    let slice2 = &*vec2;

    // Trait Objects
    let box_trait = (box 0_isize) as Box<Trait1>;
    let ref_trait = &0_isize as &Trait1;
    let mut mut_int1 = 0_isize;
    let mut_ref_trait = (&mut mut_int1) as &mut Trait1;

    let generic_box_trait = (box 0_isize) as Box<Trait2<i32, mod1::Struct2>>;
    let generic_ref_trait  = (&0_isize) as &Trait2<Struct1, Struct1>;

    let mut generic_mut_ref_trait_impl = 0_isize;
    let generic_mut_ref_trait = (&mut generic_mut_ref_trait_impl) as
        &mut Trait2<mod1::mod2::Struct3, GenericStruct<usize, isize>>;

    // Bare Functions
    let rust_fn = (rust_fn, 0_usize);
    let extern_c_fn = (extern_c_fn, 0_usize);
    let unsafe_fn = (unsafe_fn, 0_usize);
    let extern_stdcall_fn = (extern_stdcall_fn, 0_usize);

    let rust_fn_with_return_value = (rust_fn_with_return_value, 0_usize);
    let extern_c_fn_with_return_value = (extern_c_fn_with_return_value, 0_usize);
    let unsafe_fn_with_return_value = (unsafe_fn_with_return_value, 0_usize);
    let extern_stdcall_fn_with_return_value = (extern_stdcall_fn_with_return_value, 0_usize);

    let generic_function_int = (generic_function::<isize>, 0_usize);
    let generic_function_struct3 = (generic_function::<mod1::mod2::Struct3>, 0_usize);

    let variadic_function = (printf, 0_usize);

    // Closures
    // I (mw) am a bit unclear about the current state of closures, their
    // various forms (boxed, unboxed, proc, capture-by-ref, by-val, once) and
    // how that maps to rustc's internal representation of these forms.
    // Once closures have reached their 1.0 form, the tests below should
    // probably be expanded.
    let closure1 = (|x:isize| {}, 0_usize);
    let closure2 = (|x:i8, y: f32| { (x as f32) + y }, 0_usize);

    zzz(); // #break
}

#[inline(never)]
fn zzz() { () }
