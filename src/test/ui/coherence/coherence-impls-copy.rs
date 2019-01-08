// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]
#![feature(optin_builtin_traits)]

use std::marker::Copy;

impl Copy for i32 {}
//[old]~^ ERROR conflicting implementations of trait `std::marker::Copy` for type `i32`:
//[old]~| ERROR only traits defined in the current crate can be implemented for arbitrary types
//[re]~^^^ ERROR E0119
//[re]~| ERROR E0117
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
//[old]~^ ERROR the trait `Copy` may not be implemented for this type
//[re]~^^ ERROR E0206
impl Clone for MyType { fn clone(&self) -> Self { *self } }

impl Copy for (MyType, MyType) {}
//[old]~^ ERROR the trait `Copy` may not be implemented for this type
//[old]~| ERROR only traits defined in the current crate can be implemented for arbitrary types
//[re]~^^^ ERROR E0206
//[re]~| ERROR E0117
impl Copy for &'static NotSync {}
//[old]~^ ERROR conflicting implementations of trait `std::marker::Copy` for type `&NotSync`:
//[re]~^^  ERROR E0119
impl Copy for [MyType] {}
//[old]~^ ERROR the trait `Copy` may not be implemented for this type
//[old]~| ERROR only traits defined in the current crate can be implemented for arbitrary types
//[re]~^^^ ERROR E0206
//[re]~| ERROR E0117
impl Copy for &'static [NotSync] {}
//[old]~^ ERROR conflicting implementations of trait `std::marker::Copy` for type `&[NotSync]`:
//[old]~| ERROR only traits defined in the current crate can be implemented for arbitrary types
//[re]~^^^ ERROR E0119
//[re]~| ERROR E0117
fn main() {
}
