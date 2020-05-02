#![warn(clippy::new_ret_no_self)]
#![allow(dead_code)]

fn main() {}

trait R {
    type Item;
}

trait Q {
    type Item;
    type Item2;
}

struct S;

impl R for S {
    type Item = Self;
}

impl S {
    // should not trigger the lint
    pub fn new() -> impl R<Item = Self> {
        S
    }
}

struct S2;

impl R for S2 {
    type Item = Self;
}

impl S2 {
    // should not trigger the lint
    pub fn new(_: String) -> impl R<Item = Self> {
        S2
    }
}

struct S3;

impl R for S3 {
    type Item = u32;
}

impl S3 {
    // should trigger the lint
    pub fn new(_: String) -> impl R<Item = u32> {
        S3
    }
}

struct S4;

impl Q for S4 {
    type Item = u32;
    type Item2 = Self;
}

impl S4 {
    // should not trigger the lint
    pub fn new(_: String) -> impl Q<Item = u32, Item2 = Self> {
        S4
    }
}

struct T;

impl T {
    // should not trigger lint
    pub fn new() -> Self {
        unimplemented!();
    }
}

struct U;

impl U {
    // should trigger lint
    pub fn new() -> u32 {
        unimplemented!();
    }
}

struct V;

impl V {
    // should trigger lint
    pub fn new(_: String) -> u32 {
        unimplemented!();
    }
}

struct TupleReturnerOk;

impl TupleReturnerOk {
    // should not trigger lint
    pub fn new() -> (Self, u32) {
        unimplemented!();
    }
}

struct TupleReturnerOk2;

impl TupleReturnerOk2 {
    // should not trigger lint (it doesn't matter which element in the tuple is Self)
    pub fn new() -> (u32, Self) {
        unimplemented!();
    }
}

struct TupleReturnerOk3;

impl TupleReturnerOk3 {
    // should not trigger lint (tuple can contain multiple Self)
    pub fn new() -> (Self, Self) {
        unimplemented!();
    }
}

struct TupleReturnerBad;

impl TupleReturnerBad {
    // should trigger lint
    pub fn new() -> (u32, u32) {
        unimplemented!();
    }
}

struct MutPointerReturnerOk;

impl MutPointerReturnerOk {
    // should not trigger lint
    pub fn new() -> *mut Self {
        unimplemented!();
    }
}

struct MutPointerReturnerOk2;

impl MutPointerReturnerOk2 {
    // should not trigger lint
    pub fn new() -> *const Self {
        unimplemented!();
    }
}

struct MutPointerReturnerBad;

impl MutPointerReturnerBad {
    // should trigger lint
    pub fn new() -> *mut V {
        unimplemented!();
    }
}

struct GenericReturnerOk;

impl GenericReturnerOk {
    // should not trigger lint
    pub fn new() -> Option<Self> {
        unimplemented!();
    }
}

struct GenericReturnerBad;

impl GenericReturnerBad {
    // should trigger lint
    pub fn new() -> Option<u32> {
        unimplemented!();
    }
}

struct NestedReturnerOk;

impl NestedReturnerOk {
    // should not trigger lint
    pub fn new() -> (Option<Self>, u32) {
        unimplemented!();
    }
}

struct NestedReturnerOk2;

impl NestedReturnerOk2 {
    // should not trigger lint
    pub fn new() -> ((Self, u32), u32) {
        unimplemented!();
    }
}

struct NestedReturnerOk3;

impl NestedReturnerOk3 {
    // should not trigger lint
    pub fn new() -> Option<(Self, u32)> {
        unimplemented!();
    }
}

struct WithLifetime<'a> {
    cat: &'a str,
}

impl<'a> WithLifetime<'a> {
    // should not trigger the lint, because the lifetimes are different
    pub fn new<'b: 'a>(s: &'b str) -> WithLifetime<'b> {
        unimplemented!();
    }
}
