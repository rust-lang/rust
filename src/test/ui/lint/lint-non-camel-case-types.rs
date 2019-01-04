#![forbid(non_camel_case_types)]
#![allow(dead_code)]

struct ONE_TWO_THREE;
//~^ ERROR type `ONE_TWO_THREE` should have a camel case name

struct foo { //~ ERROR type `foo` should have a camel case name
    bar: isize,
}

enum foo2 { //~ ERROR type `foo2` should have a camel case name
    Bar
}

struct foo3 { //~ ERROR type `foo3` should have a camel case name
    bar: isize
}

type foo4 = isize; //~ ERROR type `foo4` should have a camel case name

enum Foo5 {
    bar //~ ERROR variant `bar` should have a camel case name
}

trait foo6 { //~ ERROR trait `foo6` should have a camel case name
    fn dummy(&self) { }
}

fn f<ty>(_: ty) {} //~ ERROR type parameter `ty` should have a camel case name

#[repr(C)]
struct foo7 {
    bar: isize,
}

struct X86_64;

struct X86__64; //~ ERROR type `X86__64` should have a camel case name

struct Abc_123; //~ ERROR type `Abc_123` should have a camel case name

struct A1_b2_c3; //~ ERROR type `A1_b2_c3` should have a camel case name

fn main() { }
