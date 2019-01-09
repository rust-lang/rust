#![forbid(non_camel_case_types)]
#![allow(dead_code)]

struct ONE_TWO_THREE;
//~^ ERROR type `ONE_TWO_THREE` should have a camel case name such as `OneTwoThree`

struct foo { //~ ERROR type `foo` should have a camel case name such as `Foo`
    bar: isize,
}

enum foo2 { //~ ERROR type `foo2` should have a camel case name such as `Foo2`
    Bar
}

struct foo3 { //~ ERROR type `foo3` should have a camel case name such as `Foo3`
    bar: isize
}

type foo4 = isize; //~ ERROR type `foo4` should have a camel case name such as `Foo4`

enum Foo5 {
    bar //~ ERROR variant `bar` should have a camel case name such as `Bar`
}

trait foo6 { //~ ERROR trait `foo6` should have a camel case name such as `Foo6`
    fn dummy(&self) { }
}

fn f<ty>(_: ty) {} //~ ERROR type parameter `ty` should have a camel case name such as `Ty`

#[repr(C)]
struct foo7 {
    bar: isize,
}

struct X86_64;

struct X86__64; //~ ERROR type `X86__64` should have a camel case name such as `X86_64`

struct Abc_123; //~ ERROR type `Abc_123` should have a camel case name such as `Abc123`

struct A1_b2_c3; //~ ERROR type `A1_b2_c3` should have a camel case name such as `A1B2C3`

fn main() { }
