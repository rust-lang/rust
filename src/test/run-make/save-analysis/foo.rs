// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![ crate_name = "test" ]
#![allow(unstable)]
#![feature(box_syntax)]

extern crate graphviz;
// A simple rust project

extern crate "flate" as myflate;

use std::collections::{HashMap,HashSet};
use std::cell::RefCell;
use std::old_io::stdio::println;


use sub::sub2 as msalias;
use sub::sub2;
use sub::sub2::nested_struct as sub_struct;
use std::num::Float;
use std::num::cast;
use std::num::{from_int,from_i8,from_i32};

use std::mem::size_of;

static uni: &'static str = "Les Miséééééééérables";
static yy: usize = 25;

static bob: Option<&'static [isize]> = None;

// buglink test - see issue #1337.

fn test_alias<I: Iterator>(i: Option<<I as Iterator>::Item>) {
    let s = sub_struct{ field2: 45u32, };

    // import tests
    fn foo(x: &Float) {}
    let _: Option<u8> = from_i32(45);

    let x = 42_usize;

    myflate::deflate_bytes(&[]);

    let x = (3, 4_usize);
    let y = x.1;
}

struct TupStruct(int, int, Box<str>);

fn test_tup_struct(x: TupStruct) -> int {
    x.1
}

mod sub {
    pub mod sub2 {
        use std::old_io::stdio::println;
        pub mod sub3 {
            use std::old_io::stdio::println;
            pub fn hello() {
                println("hello from module 3");
            }
        }
        pub fn hello() {
            println("hello from a module");
        }

        pub struct nested_struct {
            pub field2: u32,
        }

        pub enum nested_enum {
            Nest2 = 2,
            Nest3 = 3
        }
    }
}

pub mod SameDir;
pub mod SubDir;

#[path = "SameDir3.rs"]
pub mod SameDir2;

struct nofields;

#[derive(Clone)]
struct some_fields {
    field1: u32,
}

type SF = some_fields;

trait SuperTrait {
    fn dummy(&self) { }
}

trait SomeTrait: SuperTrait {
    fn Method(&self, x: u32) -> u32;

    fn prov(&self, x: u32) -> u32 {
        println(&x.to_string());
        42
    }
    fn provided_method(&self) -> u32 {
        42
    }
}

trait SubTrait: SomeTrait {
    fn stat2(x: &Self) -> u32 {
        32
    }
}

impl SomeTrait for some_fields {
    fn Method(&self, x: u32) -> u32 {
        println(&x.to_string());
        self.field1
    }
}

impl SuperTrait for some_fields {
}

impl SubTrait for some_fields {}

impl some_fields {
    fn stat(x: u32) -> u32 {
        println(&x.to_string());
        42
    }
    fn stat2(x: &some_fields) -> u32 {
        42
    }

    fn align_to<T>(&mut self) {
    }

    fn test(&mut self) {
        self.align_to::<bool>();
    }
}

impl SuperTrait for nofields {
}
impl SomeTrait for nofields {
    fn Method(&self, x: u32) -> u32 {
        self.Method(x);
        43
    }

    fn provided_method(&self) -> u32 {
        21
    }
}

impl SubTrait for nofields {}

impl SuperTrait for (Box<nofields>, Box<some_fields>) {}

fn f_with_params<T: SomeTrait>(x: &T) {
    x.Method(41);
}

type MyType = Box<some_fields>;

enum SomeEnum<'a> {
    Ints(isize, isize),
    Floats(f64, f64),
    Strings(&'a str, &'a str, &'a str),
    MyTypes(MyType, MyType)
}

#[derive(Copy)]
enum SomeOtherEnum {
    SomeConst1,
    SomeConst2,
    SomeConst3
}

enum SomeStructEnum {
    EnumStruct{a:isize, b:isize},
    EnumStruct2{f1:MyType, f2:MyType},
    EnumStruct3{f1:MyType, f2:MyType, f3:SomeEnum<'static>}
}

fn matchSomeEnum(val: SomeEnum) {
    match val {
        SomeEnum::Ints(int1, int2) => { println(&(int1+int2).to_string()); }
        SomeEnum::Floats(float1, float2) => { println(&(float2*float1).to_string()); }
        SomeEnum::Strings(_, _, s3) => { println(s3); }
        SomeEnum::MyTypes(mt1, mt2) => {
            println(&(mt1.field1 - mt2.field1).to_string());
        }
    }
}

fn matchSomeStructEnum(se: SomeStructEnum) {
    match se {
        SomeStructEnum::EnumStruct{a:a, ..} => println(&a.to_string()),
        SomeStructEnum::EnumStruct2{f1:f1, f2:f_2} => println(&f_2.field1.to_string()),
        SomeStructEnum::EnumStruct3{f1, ..} => println(&f1.field1.to_string()),
    }
}


fn matchSomeStructEnum2(se: SomeStructEnum) {
    use SomeStructEnum::*;
    match se {
        EnumStruct{a: ref aaa, ..} => println(&aaa.to_string()),
        EnumStruct2{f1, f2: f2} => println(&f1.field1.to_string()),
        EnumStruct3{f1, f3: SomeEnum::Ints(_, _), f2} => println(&f1.field1.to_string()),
        _ => {},
    }
}

fn matchSomeOtherEnum(val: SomeOtherEnum) {
    use SomeOtherEnum::{SomeConst2, SomeConst3};
    match val {
        SomeOtherEnum::SomeConst1 => { println("I'm const1."); }
        SomeConst2 | SomeConst3 => { println("I'm const2 or const3."); }
    }
}

fn hello<X: SomeTrait>((z, a) : (u32, String), ex: X) {
    SameDir2::hello(43);

    println(&yy.to_string());
    let (x, y): (u32, u32) = (5, 3);
    println(&x.to_string());
    println(&z.to_string());
    let x: u32 = x;
    println(&x.to_string());
    let x = "hello";
    println(x);

    let x = 32.0f32;
    let _ = (x + ((x * x) + 1.0).sqrt()).ln();

    let s: Box<SomeTrait> = box some_fields {field1: 43};
    let s2: Box<some_fields> =  box some_fields {field1: 43};
    let s3 = box nofields;

    s.Method(43);
    s3.Method(43);
    s2.Method(43);

    ex.prov(43);

    let y: u32 = 56;
    // static method on struct
    let r = some_fields::stat(y);
    // trait static method, calls override
    let r = SubTrait::stat2(&*s2);
    // trait static method, calls default
    let r = SubTrait::stat2(&*s3);

    let s4 = s3 as Box<SomeTrait>;
    s4.Method(43);

    s4.provided_method();
    s2.prov(45);

    let closure = |x: u32, s: &SomeTrait| {
        s.Method(23);
        return x + y;
    };

    let z = closure(10, &*s);
}

pub struct blah {
    used_link_args: RefCell<[&'static str; 0]>,
}

fn main() { // foo
    let s = box some_fields {field1: 43};
    hello((43, "a".to_string()), *s);
    sub::sub2::hello();
    sub2::sub3::hello();

    let h = sub2::sub3::hello;
    h();

    // utf8 chars
    let ut = "Les Miséééééééérables";

    // For some reason, this pattern of macro_rules foiled our generated code
    // avoiding strategy.
    macro_rules! variable_str(($name:expr) => (
        some_fields {
            field1: $name,
        }
    ));
    let vs = variable_str!(32);

    let mut candidates: RefCell<HashMap<&'static str, &'static str>> = RefCell::new(HashMap::new());
    let _ = blah {
        used_link_args: RefCell::new([]),
    };
    let s1 = nofields;
    let s2 = SF { field1: 55};
    let s3: some_fields = some_fields{ field1: 55};
    let s4: msalias::nested_struct = sub::sub2::nested_struct{ field2: 55};
    let s4: msalias::nested_struct = sub2::nested_struct{ field2: 55};
    println(&s2.field1.to_string());
    let s5: MyType = box some_fields{ field1: 55};
    let s = SameDir::SameStruct{name: "Bob".to_string()};
    let s = SubDir::SubStruct{name:"Bob".to_string()};
    let s6: SomeEnum = SomeEnum::MyTypes(box s2.clone(), s5);
    let s7: SomeEnum = SomeEnum::Strings("one", "two", "three");
    matchSomeEnum(s6);
    matchSomeEnum(s7);
    let s8: SomeOtherEnum = SomeOtherEnum::SomeConst2;
    matchSomeOtherEnum(s8);
    let s9: SomeStructEnum =
        SomeStructEnum::EnumStruct2{f1: box some_fields{field1:10}, f2: box s2};
    matchSomeStructEnum(s9);
}

impl Iterator for nofields {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        panic!()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        panic!()
    }
}
