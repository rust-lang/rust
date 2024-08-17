#![forbid(non_camel_case_types)]
#![allow(dead_code)]

struct ONE_TWO_THREE;
//~^ ERROR type `ONE_TWO_THREE` should have an upper camel case name

struct foo {
    //~^ ERROR type `foo` should have an upper camel case name
    bar: isize,
}

enum foo2 {
    //~^ ERROR type `foo2` should have an upper camel case name
    Bar,
}

struct foo3 {
    //~^ ERROR type `foo3` should have an upper camel case name
    bar: isize,
}

type foo4 = isize; //~ ERROR type `foo4` should have an upper camel case name

enum Foo5 {
    bar, //~ ERROR variant `bar` should have an upper camel case name
}

trait foo6 {
    //~^ ERROR trait `foo6` should have an upper camel case name
    type foo7; //~ ERROR associated type `foo7` should have an upper camel case name
    fn dummy(&self) {}
}

fn f<ty>(_: ty) {} //~ ERROR type parameter `ty` should have an upper camel case name

#[repr(C)]
struct foo7 {
    bar: isize,
}

struct StructRenamed;
use StructRenamed as struct_renamed; //~ ERROR renamed type `struct_renamed` should have an upper camel case name

enum EnumRenamed {
    VariantRenamed,
}
use EnumRenamed as enum_renamed; //~ ERROR renamed type `enum_renamed` should have an upper camel case name
use EnumRenamed::VariantRenamed as variant_renamed; //~ ERROR renamed variant `variant_renamed` should have an upper camel case name

type TyAliasRenamed = isize;
use TyAliasRenamed as ty_alias_renamed; //~ ERROR renamed type `ty_alias_renamed` should have an upper camel case name

use std::fmt::Display as display; //~ERROR renamed trait `display` should have an upper camel case name

fn main() {}
