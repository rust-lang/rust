//@ check-fail
//@ edition: 2024

#![allow(incomplete_features, internal_features)]
#![feature(sized_hierarchy)]
#![feature(coroutines, extern_types, f16, never_type, unsized_fn_params)]

use std::fmt::Debug;
use std::marker::{MetaSized, PointeeSized};

// This test checks that `Sized` and `MetaSized` are automatically implemented appropriately.

fn needs_sized<T: Sized>() { }
fn takes_sized<T: Sized>(_t: T) { }

fn needs_metasized<T: MetaSized>() { }
fn takes_metasized<T: MetaSized>(_t: T) { }

fn needs_pointeesized<T: PointeeSized>() { }
fn takes_pointeesized<T: PointeeSized>(_t: T) { }

fn main() {
    // `bool`
    needs_sized::<bool>();
    needs_metasized::<bool>();
    needs_pointeesized::<bool>();

    // `char`
    needs_sized::<char>();
    needs_metasized::<char>();
    needs_pointeesized::<char>();

    // `i8`
    needs_sized::<i8>();
    needs_metasized::<i8>();
    needs_pointeesized::<i8>();

    // `i16`
    needs_sized::<i16>();
    needs_metasized::<i16>();
    needs_pointeesized::<i16>();

    // `i32`
    needs_sized::<i32>();
    needs_metasized::<i32>();
    needs_pointeesized::<i32>();

    // `i64`
    needs_sized::<i64>();
    needs_metasized::<i64>();
    needs_pointeesized::<i64>();

    // `i128`
    needs_sized::<i128>();
    needs_metasized::<i128>();
    needs_pointeesized::<i128>();

    // `u8`
    needs_sized::<u8>();
    needs_metasized::<u8>();
    needs_pointeesized::<u8>();

    // `u16`
    needs_sized::<u16>();
    needs_metasized::<u16>();
    needs_pointeesized::<u16>();

    // `u32`
    needs_sized::<u32>();
    needs_metasized::<u32>();
    needs_pointeesized::<u32>();

    // `u64`
    needs_sized::<u64>();
    needs_metasized::<u64>();
    needs_pointeesized::<u64>();

    // `u128`
    needs_sized::<u128>();
    needs_metasized::<u128>();
    needs_pointeesized::<u128>();

    // `f16`
    needs_sized::<f16>();
    needs_metasized::<f16>();
    needs_pointeesized::<f16>();

    // `f32`
    needs_sized::<f32>();
    needs_metasized::<f32>();
    needs_pointeesized::<f32>();

    // `f64`
    needs_sized::<f64>();
    needs_metasized::<f64>();
    needs_pointeesized::<f64>();

    // `*const`
    needs_sized::<*const u8>();
    needs_metasized::<*const u8>();
    needs_pointeesized::<*const u8>();

    // `*mut`
    needs_sized::<*mut u8>();
    needs_metasized::<*mut u8>();
    needs_pointeesized::<*mut u8>();

    // `&`
    needs_sized::<&u8>();
    needs_metasized::<&u8>();
    needs_pointeesized::<&u8>();

    // `&mut`
    needs_sized::<&mut u8>();
    needs_metasized::<&mut u8>();
    needs_pointeesized::<&mut u8>();

    // fn-def
    fn foo(x: u8) -> u8 { x }
    takes_sized(foo);
    takes_metasized(foo);
    takes_pointeesized(foo);

    // fn-ptr
    takes_sized::<fn(u8) -> u8>(foo);
    takes_metasized::<fn(u8) -> u8>(foo);
    takes_pointeesized::<fn(u8) -> u8>(foo);

    // `[T; x]`
    needs_sized::<[u8; 1]>();
    needs_metasized::<[u8; 1]>();
    needs_pointeesized::<[u8; 1]>();

    // `|a| { a }`
    takes_sized(|a| { a });
    takes_metasized(|a| { a });
    takes_pointeesized(|a| { a });

    // `async |a| { a }`
    takes_sized(async |a| { a });
    takes_metasized(async |a| { a });
    takes_pointeesized(async |a| { a });

    // `|a| { yield a }`
    takes_sized(#[coroutine] |a| { yield a });
    takes_metasized(#[coroutine] |a| { yield a });
    takes_pointeesized(#[coroutine] |a| { yield a });

    // `!`
    needs_sized::<!>();
    needs_metasized::<!>();
    needs_pointeesized::<!>();

    // `str`
    needs_sized::<str>();
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
    needs_metasized::<str>();
    needs_pointeesized::<str>();

    // `[T]`
    needs_sized::<[u8]>();
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    needs_metasized::<[u8]>();
    needs_pointeesized::<[u8]>();

    // `dyn Debug`
    needs_sized::<dyn Debug>();
    //~^ ERROR the size for values of type `dyn Debug` cannot be known at compilation time
    needs_metasized::<dyn Debug>();
    needs_pointeesized::<dyn Debug>();

    // `extern type`
    unsafe extern "C" {
        type Foo;
    }
    needs_sized::<Foo>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known at compilation time
    needs_metasized::<Foo>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known
    needs_pointeesized::<Foo>();

    // empty tuple
    needs_sized::<()>();
    needs_metasized::<()>();
    needs_pointeesized::<()>();

    // tuple w/ all elements sized
    needs_sized::<(u32, u32)>();
    needs_metasized::<(u32, u32)>();
    needs_pointeesized::<(u32, u32)>();

    // tuple w/ all elements metasized
    needs_sized::<([u8], [u8])>();
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    needs_metasized::<([u8], [u8])>();
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    needs_pointeesized::<([u8], [u8])>();
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time

    // tuple w/ all elements pointeesized
    needs_sized::<(Foo, Foo)>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known at compilation time
    needs_metasized::<(Foo, Foo)>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known at compilation time
    //~| ERROR the size for values of type `main::Foo` cannot be known
    needs_pointeesized::<(Foo, Foo)>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known at compilation time

    // tuple w/ last element metasized
    needs_sized::<(u32, [u8])>();
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    needs_metasized::<(u32, [u8])>();
    needs_pointeesized::<(u32, [u8])>();

    // tuple w/ last element pointeesized
    needs_sized::<(u32, Foo)>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known at compilation time
    needs_metasized::<(u32, Foo)>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known
    needs_pointeesized::<(u32, Foo)>();

    // struct w/ no fields
    struct StructEmpty {}
    needs_sized::<StructEmpty>();
    needs_metasized::<StructEmpty>();
    needs_pointeesized::<StructEmpty>();

    // struct w/ all fields sized
    struct StructAllFieldsSized { x: u32, y: u32 }
    needs_sized::<StructAllFieldsSized>();
    needs_metasized::<StructAllFieldsSized>();
    needs_pointeesized::<StructAllFieldsSized>();

    // struct w/ all fields metasized
    struct StructAllFieldsMetaSized { x: [u8], y: [u8] }
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    needs_sized::<StructAllFieldsMetaSized>();
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    needs_metasized::<StructAllFieldsMetaSized>();
    needs_pointeesized::<StructAllFieldsMetaSized>();

    // struct w/ all fields unsized
    struct StructAllFieldsUnsized { x: Foo, y: Foo }
    //~^ ERROR the size for values of type `main::Foo` cannot be known at compilation time
    needs_sized::<StructAllFieldsUnsized>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known at compilation time
    needs_metasized::<StructAllFieldsUnsized>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known
    needs_pointeesized::<StructAllFieldsUnsized>();

    // struct w/ last fields metasized
    struct StructLastFieldMetaSized { x: u32, y: [u8] }
    needs_sized::<StructLastFieldMetaSized>();
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    needs_metasized::<StructLastFieldMetaSized>();
    needs_pointeesized::<StructLastFieldMetaSized>();

    // struct w/ last fields unsized
    struct StructLastFieldUnsized { x: u32, y: Foo }
    needs_sized::<StructLastFieldUnsized>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known at compilation time
    needs_metasized::<StructLastFieldUnsized>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known
    needs_pointeesized::<StructLastFieldUnsized>();

    // enum w/ no fields
    enum EnumEmpty {}
    needs_sized::<EnumEmpty>();
    needs_metasized::<EnumEmpty>();
    needs_pointeesized::<EnumEmpty>();

    // enum w/ all variant fields sized
    enum EnumAllFieldsSized { Qux { x: u32, y: u32 } }
    needs_sized::<StructAllFieldsSized>();
    needs_metasized::<StructAllFieldsSized>();
    needs_pointeesized::<StructAllFieldsSized>();

    // enum w/ all variant fields metasized
    enum EnumAllFieldsMetaSized { Qux { x: [u8], y: [u8] } }
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    needs_sized::<EnumAllFieldsMetaSized>();
    needs_metasized::<EnumAllFieldsMetaSized>();
    needs_pointeesized::<EnumAllFieldsMetaSized>();

    // enum w/ all variant fields unsized
    enum EnumAllFieldsUnsized { Qux { x: Foo, y: Foo } }
    //~^ ERROR the size for values of type `main::Foo` cannot be known at compilation time
    needs_sized::<EnumAllFieldsUnsized>();
    needs_metasized::<EnumAllFieldsUnsized>();
    needs_pointeesized::<EnumAllFieldsUnsized>();

    // enum w/ last variant fields metasized
    enum EnumLastFieldMetaSized { Qux { x: u32, y: [u8] } }
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    needs_sized::<EnumLastFieldMetaSized>();
    needs_metasized::<EnumLastFieldMetaSized>();
    needs_pointeesized::<EnumLastFieldMetaSized>();

    // enum w/ last variant fields unsized
    enum EnumLastFieldUnsized { Qux { x: u32, y: Foo } }
    //~^ ERROR the size for values of type `main::Foo` cannot be known at compilation time
    needs_sized::<EnumLastFieldUnsized>();
    needs_metasized::<EnumLastFieldUnsized>();
    needs_pointeesized::<EnumLastFieldUnsized>();
}
