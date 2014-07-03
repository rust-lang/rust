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
// ignore-android: FIXME(#10381)

// compile-flags:-g
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish


// STRUCTS
// gdb-command:whatis simple_struct
// gdb-check:type = struct Struct1

// gdb-command:whatis generic_struct1
// gdb-check:type = struct GenericStruct<type-names::Mod1::Struct2, type-names::Mod1::Mod2::Struct3>

// gdb-command:whatis generic_struct2
// gdb-check:type = struct GenericStruct<type-names::Struct1, extern "fastcall" fn(int) -> uint>

// gdb-command:whatis mod_struct
// gdb-check:type = struct Struct2


// ENUMS
// gdb-command:whatis simple_enum_1
// gdb-check:type = union Enum1

// gdb-command:whatis simple_enum_2
// gdb-check:type = union Enum1

// gdb-command:whatis simple_enum_3
// gdb-check:type = union Enum2

// gdb-command:whatis generic_enum_1
// gdb-check:type = union Enum3<type-names::Mod1::Struct2>

// gdb-command:whatis generic_enum_2
// gdb-check:type = union Enum3<type-names::Struct1>


// TUPLES
// gdb-command:whatis tuple1
// gdb-check:type = struct (u32, type-names::Struct1, type-names::Mod1::Mod2::Enum3<type-names::Mod1::Struct2>)

// gdb-command:whatis tuple2
// gdb-check:type = struct ((type-names::Struct1, type-names::Mod1::Mod2::Struct3), type-names::Mod1::Enum2, char)


// BOX
// gdb-command:whatis box1
// gdb-check:type = struct (Box<f32>, i32)

// gdb-command:whatis box2
// gdb-check:type = struct (Box<type-names::Mod1::Mod2::Enum3<f32>>, i32)


// REFERENCES
// gdb-command:whatis ref1
// gdb-check:type = struct (&type-names::Struct1, i32)

// gdb-command:whatis ref2
// gdb-check:type = struct (&type-names::GenericStruct<char, type-names::Struct1>, i32)

// gdb-command:whatis mut_ref1
// gdb-check:type = struct (&mut type-names::Struct1, i32)

// gdb-command:whatis mut_ref2
// gdb-check:type = struct (&mut type-names::GenericStruct<type-names::Mod1::Enum2, f64>, i32)


// RAW POINTERS
// gdb-command:whatis mut_ptr1
// gdb-check:type = struct (*mut type-names::Struct1, int)

// gdb-command:whatis mut_ptr2
// gdb-check:type = struct (*mut int, int)

// gdb-command:whatis mut_ptr3
// gdb-check:type = struct (*mut type-names::Mod1::Mod2::Enum3<type-names::Struct1>, int)

// gdb-command:whatis const_ptr1
// gdb-check:type = struct (*const type-names::Struct1, int)

// gdb-command:whatis const_ptr2
// gdb-check:type = struct (*const int, int)

// gdb-command:whatis const_ptr3
// gdb-check:type = struct (*const type-names::Mod1::Mod2::Enum3<type-names::Struct1>, int)


// VECTORS
// gdb-command:whatis fixed_size_vec1
// gdb-check:type = struct ([type-names::Struct1, ..3], i16)

// gdb-command:whatis fixed_size_vec2
// gdb-check:type = struct ([uint, ..3], i16)

// gdb-command:whatis slice1
// gdb-check:type = struct &[uint]

// gdb-command:whatis slice2
// gdb-check:type = struct &[type-names::Mod1::Enum2]


// TRAITS
// gdb-command:whatis box_trait
// gdb-check:type = struct Box<Trait1>

// gdb-command:whatis ref_trait
// gdb-check:type = struct &Trait1

// gdb-command:whatis mut_ref_trait
// gdb-check:type = struct &mut Trait1

// gdb-command:whatis generic_box_trait
// gdb-check:type = struct Box<Trait2<i32, type-names::Mod1::Struct2>>

// gdb-command:whatis generic_ref_trait
// gdb-check:type = struct &Trait2<type-names::Struct1, type-names::Struct1>

// gdb-command:whatis generic_mut_ref_trait
// gdb-check:type = struct &mut Trait2<type-names::Mod1::Mod2::Struct3, type-names::GenericStruct<uint, int>>


// BARE FUNCTIONS
// gdb-command:whatis rust_fn
// gdb-check:type = struct (fn(core::option::Option<int>, core::option::Option<&type-names::Mod1::Struct2>), uint)

// gdb-command:whatis extern_c_fn
// gdb-check:type = struct (extern "C" fn(int), uint)

// gdb-command:whatis unsafe_fn
// gdb-check:type = struct (unsafe fn(core::result::Result<char, f64>), uint)

// gdb-command:whatis extern_stdcall_fn
// gdb-check:type = struct (extern "stdcall" fn(), uint)

// gdb-command:whatis rust_fn_with_return_value
// gdb-check:type = struct (fn(f64) -> uint, uint)

// gdb-command:whatis extern_c_fn_with_return_value
// gdb-check:type = struct (extern "C" fn() -> type-names::Struct1, uint)

// gdb-command:whatis unsafe_fn_with_return_value
// gdb-check:type = struct (unsafe fn(type-names::GenericStruct<u16, u8>) -> type-names::Mod1::Struct2, uint)

// gdb-command:whatis extern_stdcall_fn_with_return_value
// gdb-check:type = struct (extern "stdcall" fn(Box<int>) -> uint, uint)

// gdb-command:whatis generic_function_int
// gdb-check:type = struct (fn(int) -> int, uint)

// gdb-command:whatis generic_function_struct3
// gdb-check:type = struct (fn(type-names::Mod1::Mod2::Struct3) -> type-names::Mod1::Mod2::Struct3, uint)

// gdb-command:whatis variadic_function
// gdb-check:type = struct (unsafe extern "C" fn(*const u8, ...) -> int, uint)


// CLOSURES
// gdb-command:whatis some_proc
// gdb-check:type = struct (once proc(int, u8) -> (int, u8), uint)

// gdb-command:whatis stack_closure1
// gdb-check:type = struct (&mut|int|, uint)

// gdb-command:whatis stack_closure2
// gdb-check:type = struct (&mut|i8, f32| -> f32, uint)

use std::ptr;

struct Struct1;
struct GenericStruct<T1, T2>;

enum Enum1 {
    Variant1_1,
    Variant1_2(int)
}

mod Mod1 {
    pub struct Struct2;

    pub enum Enum2 {
        Variant2_1,
        Variant2_2(super::Struct1)
    }

    pub mod Mod2 {
        pub struct Struct3;

        pub enum Enum3<T> {
            Variant3_1,
            Variant3_2(T),
        }
    }
}

trait Trait1 { }
trait Trait2<T1, T2> { }

impl Trait1 for int {}
impl<T1, T2> Trait2<T1, T2> for int {}

fn rust_fn(_: Option<int>, _: Option<&Mod1::Struct2>) {}
extern "C" fn extern_c_fn(_: int) {}
unsafe fn unsafe_fn(_: Result<char, f64>) {}
extern "stdcall" fn extern_stdcall_fn() {}

fn rust_fn_with_return_value(_: f64) -> uint { 4 }
extern "C" fn extern_c_fn_with_return_value() -> Struct1 { Struct1 }
unsafe fn unsafe_fn_with_return_value(_: GenericStruct<u16, u8>) -> Mod1::Struct2 { Mod1::Struct2 }
extern "stdcall" fn extern_stdcall_fn_with_return_value(_: Box<int>) -> uint { 0 }

fn generic_function<T>(x: T) -> T { x }

extern {
    fn printf(_:*const u8, ...) -> int;
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
    let generic_struct1: GenericStruct<Mod1::Struct2, Mod1::Mod2::Struct3> = GenericStruct;
    let generic_struct2: GenericStruct<Struct1, extern "fastcall" fn(int) -> uint> = GenericStruct;
    let mod_struct = Mod1::Struct2;

    // Enums
    let simple_enum_1 = Variant1_1;
    let simple_enum_2 = Variant1_2(0);
    let simple_enum_3 = Mod1::Variant2_2(Struct1);

    let generic_enum_1: Mod1::Mod2::Enum3<Mod1::Struct2> = Mod1::Mod2::Variant3_1;
    let generic_enum_2 = Mod1::Mod2::Variant3_2(Struct1);

    // Tuples
    let tuple1 = (8u32, Struct1, Mod1::Mod2::Variant3_2(Mod1::Struct2));
    let tuple2 = ((Struct1, Mod1::Mod2::Struct3), Mod1::Variant2_1, 'x');

    // Box
    let box1 = (box 1f32, 0i32);
    let box2 = (box Mod1::Mod2::Variant3_2(1f32), 0i32);

    // References
    let ref1 = (&Struct1, 0i32);
    let ref2 = (&GenericStruct::<char, Struct1>, 0i32);

    let mut mut_struct1 = Struct1;
    let mut mut_generic_struct = GenericStruct::<Mod1::Enum2, f64>;
    let mut_ref1 = (&mut mut_struct1, 0i32);
    let mut_ref2 = (&mut mut_generic_struct, 0i32);

    // Raw Pointers
    let mut_ptr1: (*mut Struct1, int) = (ptr::mut_null(), 0);
    let mut_ptr2: (*mut int, int) = (ptr::mut_null(), 0);
    let mut_ptr3: (*mut Mod1::Mod2::Enum3<Struct1>, int) = (ptr::mut_null(), 0);

    let const_ptr1: (*const Struct1, int) = (ptr::null(), 0);
    let const_ptr2: (*const int, int) = (ptr::null(), 0);
    let const_ptr3: (*const Mod1::Mod2::Enum3<Struct1>, int) = (ptr::null(), 0);

    // Vectors
    let fixed_size_vec1 = ([Struct1, Struct1, Struct1], 0i16);
    let fixed_size_vec2 = ([0u, 1u, 2u], 0i16);

    let vec1 = vec![0u, 2u, 3u];
    let slice1 = vec1.as_slice();
    let vec2 = vec![Mod1::Variant2_2(Struct1)];
    let slice2 = vec2.as_slice();

    // Trait Objects
    let box_trait = (box 0i) as Box<Trait1>;
    let ref_trait = &0i as &Trait1;
    let mut mut_int1 = 0i;
    let mut_ref_trait = (&mut mut_int1) as &mut Trait1;

    let generic_box_trait = (box 0i) as Box<Trait2<i32, Mod1::Struct2>>;
    let generic_ref_trait  = (&0i) as &Trait2<Struct1, Struct1>;

    let mut generic_mut_ref_trait_impl = 0i;
    let generic_mut_ref_trait = (&mut generic_mut_ref_trait_impl) as
        &mut Trait2<Mod1::Mod2::Struct3, GenericStruct<uint, int>>;

    // Bare Functions
    let rust_fn = (rust_fn, 0u);
    let extern_c_fn = (extern_c_fn, 0u);
    let unsafe_fn = (unsafe_fn, 0u);
    let extern_stdcall_fn = (extern_stdcall_fn, 0u);

    let rust_fn_with_return_value = (rust_fn_with_return_value, 0u);
    let extern_c_fn_with_return_value = (extern_c_fn_with_return_value, 0u);
    let unsafe_fn_with_return_value = (unsafe_fn_with_return_value, 0u);
    let extern_stdcall_fn_with_return_value = (extern_stdcall_fn_with_return_value, 0u);

    let generic_function_int = (generic_function::<int>, 0u);
    let generic_function_struct3 = (generic_function::<Mod1::Mod2::Struct3>, 0u);

    let variadic_function = (printf, 0u);

    // Closures
    // I (mw) am a bit unclear about the current state of closures, their
    // various forms (boxed, unboxed, proc, capture-by-ref, by-val, once) and
    // how that maps to rustc's internal representation of these forms.
    // Once closures have reached their 1.0 form, the tests below should
    // probably be expanded.
    let some_proc = (proc(a:int, b:u8) (a, b), 0u);

    let stack_closure1 = (|x:int| {}, 0u);
    let stack_closure2 = (|x:i8, y: f32| { (x as f32) + y }, 0u);

    zzz();
}

#[inline(never)]
fn zzz() { () }
