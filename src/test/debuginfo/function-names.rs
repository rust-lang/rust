// Function names are formatted differently in old versions of GDB
// min-gdb-version: 10.1

// compile-flags:-g

// === GDB TESTS ===================================================================================

// Top-level function
// gdb-command:info functions -q function_names::main
// gdb-check:[...]static fn function_names::main();
// gdb-command:info functions -q function_names::generic_func<*
// gdb-check:[...]static fn function_names::generic_func<i32>(i32) -> i32;

// Implementations
// gdb-command:info functions -q function_names::.*::impl_function.*
// gdb-check:[...]static fn function_names::GenericStruct<i32, i32>::impl_function<i32, i32>();
// gdb-check:[...]static fn function_names::Mod1::TestStruct2::impl_function();
// gdb-check:[...]static fn function_names::TestStruct1::impl_function();

// Trait implementations
// gdb-command:info functions -q function_names::.*::trait_function.*
// gdb-check:[...]static fn function_names::Mod1::{impl#1}::trait_function();
// gdb-check:[...]static fn function_names::{impl#1}::trait_function();
// gdb-check:[...]static fn function_names::{impl#3}::trait_function<i32>();
// gdb-check:[...]static fn function_names::{impl#5}::trait_function3<function_names::TestStruct1>();
// gdb-check:[...]static fn function_names::{impl#6}::trait_function<i32, 1>();

// Closure
// gdb-command:info functions -q function_names::.*::{closure.*
// gdb-check:[...]static fn function_names::generic_func::{closure#0}<i32>(*mut function_names::generic_func::{closure#0});
// gdb-check:[...]static fn function_names::main::{closure#0}(*mut function_names::main::{closure#0});
// gdb-check:[...]static fn function_names::{impl#2}::impl_function::{closure#0}<i32, i32>(*mut function_names::{impl#2}::impl_function::{closure#0});

// Generator
// Generators don't seem to appear in GDB's symbol table.

// Const generic parameter
// gdb-command:info functions -q function_names::const_generic_fn.*
// gdb-check:[...]static fn function_names::const_generic_fn_bool<false>();
// gdb-check:[...]static fn function_names::const_generic_fn_non_int<{CONST#fe3cfa0214ac55c7}>();
// gdb-check:[...]static fn function_names::const_generic_fn_signed_int<-7>();
// gdb-check:[...]static fn function_names::const_generic_fn_unsigned_int<14>();

// === CDB TESTS ===================================================================================

// Top-level function
// cdb-command:x a!function_names::main
// cdb-check:[...] a!function_names::main (void)
// cdb-command:x a!function_names::generic_func<*
// cdb-check:[...] a!function_names::generic_func<i32> (int)

// Implementations
// cdb-command:x a!function_names::*::impl_function*
// cdb-check:[...] a!function_names::Mod1::TestStruct2::impl_function (void)
// cdb-check:[...] a!function_names::TestStruct1::impl_function (void)
// cdb-check:[...] a!function_names::GenericStruct<i32,i32>::impl_function<i32,i32> (void)

// Trait implementations
// cdb-command:x a!function_names::*::trait_function*
// cdb-check:[...] a!function_names::impl$3::trait_function<i32> (void)
// cdb-check:[...] a!function_names::impl$6::trait_function<i32,1> (void)
// cdb-check:[...] a!function_names::impl$1::trait_function (void)
// cdb-check:[...] a!function_names::impl$5::trait_function3<function_names::TestStruct1> (void)
// cdb-check:[...] a!function_names::Mod1::impl$1::trait_function (void)

// Closure
// cdb-command:x a!function_names::*::closure*
// cdb-check:[...] a!function_names::impl$2::impl_function::closure$0<i32,i32> (void)
// cdb-check:[...] a!function_names::main::closure$0 (void)
// cdb-check:[...] a!function_names::generic_func::closure$0<i32> (void)

// Generator
// cdb-command:x a!function_names::*::generator*
// cdb-check:[...] a!function_names::main::generator$1 (void)

// Const generic parameter
// cdb-command:x a!function_names::const_generic_fn*
// cdb-check:[...] a!function_names::const_generic_fn_bool<false> (void)
// cdb-check:[...] a!function_names::const_generic_fn_non_int<CONST$fe3cfa0214ac55c7> (void)
// cdb-check:[...] a!function_names::const_generic_fn_unsigned_int<14> (void)
// cdb-check:[...] a!function_names::const_generic_fn_signed_int<-7> (void)

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]
#![feature(adt_const_params, generators, generator_trait)]
#![allow(incomplete_features)]

use Mod1::TestTrait2;
use std::ops::Generator;
use std::pin::Pin;

fn main() {
    // Implementations
    TestStruct1::impl_function();
    Mod1::TestStruct2::impl_function();
    GenericStruct::<i32, i32>::impl_function();

    // Trait implementations
    TestStruct1::trait_function();
    Mod1::TestStruct2::trait_function();
    GenericStruct::<i32, i32>::trait_function();
    GenericStruct::<[i32; 1], f32>::trait_function();
    GenericStruct::<TestStruct1, usize>::trait_function3();

    // Generic function
    let _ = generic_func(42i32);

    // Closure
    let closure = || { TestStruct1 };
    closure();

    // Generator
    let mut generator = || { yield; return; };
    Pin::new(&mut generator).resume(());

    // Const generic functions
    const_generic_fn_bool::<false>();
    const_generic_fn_non_int::<{()}>();
    const_generic_fn_signed_int::<-7>();
    const_generic_fn_unsigned_int::<14>();
}

struct TestStruct1;
trait TestTrait1 {
    fn trait_function();
}

// Implementation
impl TestStruct1 {
    pub fn impl_function() {}
}

// Implementation for a trait
impl TestTrait1 for TestStruct1 {
    fn trait_function() {}
}

// Implementation and implementation within a mod
mod Mod1 {
    pub struct TestStruct2;
    pub trait TestTrait2 {
        fn trait_function();
    }

    impl TestStruct2 {
        pub fn impl_function() {}
    }

    impl TestTrait2 for TestStruct2 {
        fn trait_function() {}
    }
}

struct GenericStruct<T1, T2>(std::marker::PhantomData<(T1, T2)>);

// Generic implementation
impl<T1, T2> GenericStruct<T1, T2> {
    pub fn impl_function() {
        // Closure in a generic implementation
        let closure = || { TestStruct1 };
        closure();
    }
}

// Generic trait implementation
impl<T> TestTrait1 for GenericStruct<T, i32> {
    fn trait_function() {}
}

// Implementation based on associated type
trait TestTrait3 {
    type AssocType;
    fn trait_function3();
}
impl TestTrait3 for TestStruct1 {
    type AssocType = usize;
    fn trait_function3() {}
}
impl<T: TestTrait3> TestTrait3 for GenericStruct<T, T::AssocType> {
    type AssocType = T::AssocType;
    fn trait_function3() {}
}

// Generic trait implementation with const generics
impl<T, const N: usize> TestTrait1 for GenericStruct<[T; N], f32> {
    fn trait_function() {}
}

// Generic function
fn generic_func<T>(value: T) -> T {
    // Closure in a generic function
    let closure = || { TestStruct1 };
    closure();

    value
}

fn const_generic_fn_bool<const C: bool>() {}
fn const_generic_fn_non_int<const C: ()>() {}
fn const_generic_fn_signed_int<const C: i64>() {}
fn const_generic_fn_unsigned_int<const C: u32>() {}
