#![feature(custom_attribute)]
#![feature(raw_identifiers)]
#![feature(extern_types)]
#![allow(invalid_type_param_default)]
#![allow(unused_attributes)]

use r#foo as r#alias_foo;

// https://github.com/rust-lang/rustfmt/issues/3837
pub(crate) static r#break: &'static str = "foo";

fn main() {
    #[r#attr]
    r#foo::r#bar();

    let r#local = r#Struct { r#field: () };
    r#local.r#field = 1;
    r#foo.r#barr();
    let r#async = r#foo(r#local);
    r#macro!();

    if let r#sub_pat @ r#Foo(_) = r#Foo(3) {}

    match r#async {
        r#Foo | r#Bar => r#foo(),
    }
}

fn r#bar<'a, r#T>(r#x: &'a r#T) {}

mod r#foo {
    pub fn r#bar() {}
}

enum r#Foo {
    r#Bar {},
}

struct r#Struct {
    r#field: r#FieldType,
}

trait r#Trait {
    type r#Type;
}

impl r#Trait for r#Impl {
    type r#Type = r#u32;
    fn r#xxx(r#fjio: r#u32) {}
}

extern "C" {
    type r#ccc;
    static r#static_val: u32;
}

macro_rules! r#macro {
    () => {};
}

macro_rules! foo {
    ($x:expr) => {
        let r#catch = $x + 1;
        println!("{}", r#catch);
    };
}
