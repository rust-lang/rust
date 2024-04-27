#![feature(const_mut_refs)]

//@ normalize-stderr-test "\(size: ., align: .\)" -> ""
//@ normalize-stderr-test " +│ ╾─+╼" -> ""

static X: i32 = 1;
const C: i32 = 2;
static mut M: i32 = 3;

const CR: &'static mut i32 = &mut C; //~ ERROR mutable references are not allowed
//~| WARN taking a mutable

static STATIC_REF: &'static mut i32 = &mut X; //~ ERROR cannot borrow immutable static item `X` as mutable

static CONST_REF: &'static mut i32 = &mut C; //~ ERROR mutable references are not allowed
//~| WARN taking a mutable

fn main() {}
