#![feature(tool_lints)]

#![warn(clippy::new_ret_no_self)]
#![allow(dead_code, clippy::trivially_copy_pass_by_ref)]

fn main(){}

trait R {
    type Item;
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
    // should trigger the lint, but currently does not
    pub fn new(_: String) -> impl R<Item = u32> {
        S3
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
