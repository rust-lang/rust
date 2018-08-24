#![feature(optin_builtin_traits)]

use std::marker::Copy;

impl Copy for i32 {}
//~^ ERROR conflicting implementations of trait `std::marker::Copy` for type `i32`:
//~| ERROR only traits defined in the current crate can be implemented for arbitrary types

enum TestE {
  A
}

struct MyType;

struct NotSync;
impl !Sync for NotSync {}

impl Copy for TestE {}
impl Clone for TestE { fn clone(&self) -> Self { *self } }

impl Copy for MyType {}

impl Copy for &'static mut MyType {}
//~^ ERROR the trait `Copy` may not be implemented for this type
impl Clone for MyType { fn clone(&self) -> Self { *self } }

impl Copy for (MyType, MyType) {}
//~^ ERROR the trait `Copy` may not be implemented for this type
//~| ERROR only traits defined in the current crate can be implemented for arbitrary types

impl Copy for &'static NotSync {}
//~^ ERROR conflicting implementations of trait `std::marker::Copy` for type `&NotSync`:

impl Copy for [MyType] {}
//~^ ERROR the trait `Copy` may not be implemented for this type
//~| ERROR only traits defined in the current crate can be implemented for arbitrary types

impl Copy for &'static [NotSync] {}
//~^ ERROR conflicting implementations of trait `std::marker::Copy` for type `&[NotSync]`:
//~| ERROR only traits defined in the current crate can be implemented for arbitrary types

fn main() {
}
