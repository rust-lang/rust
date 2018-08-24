#![feature(optin_builtin_traits)]

use std::marker::Copy;

enum TestE {
  A
}

struct MyType;

struct NotSync;
impl !Sync for NotSync {}

impl Sized for TestE {} //~ ERROR E0322
//~^ impl of 'Sized' not allowed

impl Sized for MyType {} //~ ERROR E0322
//~^ impl of 'Sized' not allowed

impl Sized for (MyType, MyType) {} //~ ERROR E0322
//~^ impl of 'Sized' not allowed
//~| ERROR E0117

impl Sized for &'static NotSync {} //~ ERROR E0322
//~^ impl of 'Sized' not allowed

impl Sized for [MyType] {} //~ ERROR E0322
//~^ impl of 'Sized' not allowed
//~| ERROR E0117

impl Sized for &'static [NotSync] {} //~ ERROR E0322
//~^ impl of 'Sized' not allowed
//~| ERROR E0117

fn main() {
}
