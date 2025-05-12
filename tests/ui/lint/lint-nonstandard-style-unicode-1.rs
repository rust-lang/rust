#![allow(dead_code)]

#![forbid(non_camel_case_types)]

// Some scripts (e.g., hiragana) don't have a concept of
// upper/lowercase

// 1. non_camel_case_types

// Can start with non-lowercase letter
struct Θχ;
struct ヒa;

struct χa;
//~^ ERROR type `χa` should have an upper camel case name

// If there's already leading or trailing underscores, they get trimmed before checking.
// This is fine:
struct _ヒb;

// This is not:
struct __χa;
//~^ ERROR type `__χa` should have an upper camel case name

// Besides this, we cannot have two continuous underscores in the middle.

struct 对__否;
//~^ ERROR type `对__否` should have an upper camel case name

struct ヒ__χ;
//~^ ERROR type `ヒ__χ` should have an upper camel case name

// also cannot have lowercase letter next to an underscore.
// so this triggers the lint:

struct Hello_你好;
//~^ ERROR type `Hello_你好` should have an upper camel case name

struct Hello_World;
//~^ ERROR type `Hello_World` should have an upper camel case name

struct 你_ӟ;
//~^ ERROR type `你_ӟ` should have an upper camel case name

// and this is ok:

struct 你_好;

fn main() {}
