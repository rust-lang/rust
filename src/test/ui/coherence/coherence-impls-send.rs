#![feature(optin_builtin_traits)]
#![feature(overlapping_marker_traits)]

use std::marker::Copy;

enum TestE {
  A
}

struct MyType;

struct NotSync;
impl !Sync for NotSync {}

unsafe impl Send for TestE {}
unsafe impl Send for MyType {}
unsafe impl Send for (MyType, MyType) {}
//~^ ERROR E0117

unsafe impl Send for &'static NotSync {}
//~^ ERROR E0321

unsafe impl Send for [MyType] {}
//~^ ERROR E0117

unsafe impl Send for &'static [NotSync] {}
//~^ ERROR E0117

fn main() {
}
