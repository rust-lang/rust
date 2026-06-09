#![feature(negative_impls)]

use std::marker::Copy;

impl Copy for i32 {}
//~^ ERROR E0117
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
//~^ ERROR E0206
impl Clone for MyType { fn clone(&self) -> Self { *self } }

impl Copy for (MyType, MyType) {}
//~^ ERROR E0206
//~| ERROR E0117
impl Copy for &'static NotSync {}
//~^  ERROR E0119
impl Copy for [MyType] {}
//~^ ERROR E0206
//~| ERROR E0117
impl Copy for &'static [NotSync] {}
//~^ ERROR E0117
fn main() {
}
