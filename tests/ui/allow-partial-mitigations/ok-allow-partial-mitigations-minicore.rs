// ignore-tidy-linelength
//@ check-pass
//@ add-minicore
//@ edition:future
//@ revisions: default deny
//@[default] compile-flags: -Z unstable-options -Z stack-protector=all
//@[deny] compile-flags: -Z allow-partial-mitigations=!stack-protector -Z unstable-options -Z stack-protector=all

// ^ enables stack-protector for both minicore and this crate

#![crate_type = "lib"]
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;

pub fn foo() {}
