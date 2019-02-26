#![forbid(non_camel_case_types)]
#![allow(dead_code)]

struct ONE_TWO_THREE;
//~^ ERROR type `ONE_TWO_THREE` should have an upper camel case name

struct foo { //~ ERROR type `foo` should have an upper camel case name
    bar: isize,
}

enum foo2 { //~ ERROR type `foo2` should have an upper camel case name
    Bar
}

struct foo3 { //~ ERROR type `foo3` should have an upper camel case name
    bar: isize
}

type foo4 = isize; //~ ERROR type `foo4` should have an upper camel case name

enum Foo5 {
    bar //~ ERROR variant `bar` should have an upper camel case name
}

trait foo6 { //~ ERROR trait `foo6` should have an upper camel case name
    fn dummy(&self) { }
}

fn f<ty>(_: ty) {} //~ ERROR type parameter `ty` should have an upper camel case name

#[repr(C)]
struct foo7 {
    bar: isize,
}

fn main() { }
