#![no_std]
#![allow(unused_variables)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![deny(dead_code)]

#![crate_type="lib"]

pub use foo2::Bar2;

mod foo {
    pub struct Bar; //~ ERROR: struct `Bar` is never constructed
}

mod foo2 {
    pub struct Bar2;
}

pub static pub_static: isize = 0;
static priv_static: isize = 0; //~ ERROR: static `priv_static` is never used
const used_static: isize = 0;
pub static used_static2: isize = used_static;
const USED_STATIC: isize = 0;
const STATIC_USED_IN_ENUM_DISCRIMINANT: isize = 10;

pub const pub_const: isize = 0;
const priv_const: isize = 0; //~ ERROR: constant `priv_const` is never used
const used_const: isize = 0;
pub const used_const2: isize = used_const;
const USED_CONST: isize = 1;
const CONST_USED_IN_ENUM_DISCRIMINANT: isize = 11;
const CONST_USED_IN_RANGE_PATTERN: isize = 12;

pub type typ = *const UsedStruct4;
pub struct PubStruct;
struct PrivStruct; //~ ERROR: struct `PrivStruct` is never constructed
struct UsedStruct1 {
    #[allow(dead_code)]
    x: isize
}
struct UsedStruct2(#[allow(dead_code)] isize);
struct UsedStruct3;
pub struct UsedStruct4;
// this struct is never used directly, but its method is, so we don't want
// to warn it
struct SemiUsedStruct;
impl SemiUsedStruct {
    fn la_la_la() {}
}
struct StructUsedAsField;
pub struct StructUsedInEnum;
struct StructUsedInGeneric;
pub struct PubStruct2 {
    #[allow(dead_code)]
    struct_used_as_field: *const StructUsedAsField
}

pub enum pub_enum { foo1, bar1 }
pub enum pub_enum2 { a(*const StructUsedInEnum) }
pub enum pub_enum3 {
    Foo = STATIC_USED_IN_ENUM_DISCRIMINANT,
    Bar = CONST_USED_IN_ENUM_DISCRIMINANT,
}

enum priv_enum { foo2, bar2 } //~ ERROR: enum `priv_enum` is never used
enum used_enum {
    foo3,
    bar3 //~ ERROR variant `bar3` is never constructed
}

fn f<T>() {}

pub fn pub_fn() {
    used_fn();
    let used_struct1 = UsedStruct1 { x: 1 };
    let used_struct2 = UsedStruct2(1);
    let used_struct3 = UsedStruct3;
    let e = used_enum::foo3;
    SemiUsedStruct::la_la_la();

    let i = 1;
    match i {
        USED_STATIC => (),
        USED_CONST => (),
        CONST_USED_IN_RANGE_PATTERN..100 => {}
        _ => ()
    }
    f::<StructUsedInGeneric>();
}
fn priv_fn() { //~ ERROR: function `priv_fn` is never used
    let unused_struct = PrivStruct;
}
fn used_fn() {}

fn foo() { //~ ERROR: function `foo` is never used
    bar();
    let unused_enum = priv_enum::foo2;
}

fn bar() { //~ ERROR: function `bar` is never used
    foo();
}

fn baz() -> impl Copy { //~ ERROR: function `baz` is never used
    "I'm unused, too"
}

// Code with #[allow(dead_code)] should be marked live (and thus anything it
// calls is marked live)
#[allow(dead_code)]
fn g() { h(); }
fn h() {}
