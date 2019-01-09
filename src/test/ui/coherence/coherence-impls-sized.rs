// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]
#![feature(optin_builtin_traits)]

use std::marker::Copy;

enum TestE {
  A
}

struct MyType;

struct NotSync;
impl !Sync for NotSync {}

impl Sized for TestE {}
//[old]~^ ERROR E0322
//[old]~| impl of 'Sized' not allowed
//[re]~^^^ ERROR E0322

impl Sized for MyType {}
//[old]~^ ERROR E0322
//[old]~| impl of 'Sized' not allowed
//[re]~^^^ ERROR E0322

impl Sized for (MyType, MyType) {}
//[old]~^ ERROR E0322
//[old]~| impl of 'Sized' not allowed
//[old]~| ERROR E0117
//[re]~^^^^ ERROR E0322
//[re]~| ERROR E0117

impl Sized for &'static NotSync {}
//[old]~^ ERROR E0322
//[old]~| impl of 'Sized' not allowed
//[re]~^^^ ERROR E0322

impl Sized for [MyType] {}
//[old]~^ ERROR E0322
//[old]~| impl of 'Sized' not allowed
//[old]~| ERROR E0117
//[re]~^^^^ ERROR E0322
//[re]~| ERROR E0117

impl Sized for &'static [NotSync] {}
//[old]~^ ERROR E0322
//[old]~| impl of 'Sized' not allowed
//[old]~| ERROR E0117
//[re]~^^^^ ERROR E0322
//[re]~| ERROR E0117

fn main() {
}
