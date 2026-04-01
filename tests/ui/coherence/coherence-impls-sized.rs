#![feature(negative_impls)]

use std::marker::Copy;

enum TestE {
  A
}

struct MyType;

struct NotSync;
impl !Sync for NotSync {}

impl Sized for TestE {}
//~^ ERROR E0322

impl Sized for MyType {}
//~^ ERROR E0322

impl Sized for (MyType, MyType) {}
//~^ ERROR E0322
//~| ERROR E0117

impl Sized for &'static NotSync {}
//~^ ERROR E0322

impl Sized for [MyType] {}
//~^ ERROR E0322
//~| ERROR E0117

impl Sized for &'static [NotSync] {}
//~^ ERROR E0322
//~| ERROR E0117

fn main() {
}
