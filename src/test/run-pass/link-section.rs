// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(not(target_os = "macos"))]
#[link_section=".moretext"]
fn i_live_in_more_text() -> &'static str {
    "knock knock"
}

#[cfg(not(target_os = "macos"))]
#[link_section=".imm"]
static magic: uint = 42;

#[cfg(not(target_os = "macos"))]
#[link_section=".mut"]
static mut frobulator: uint = 0xdeadbeef;

#[cfg(target_os = "macos")]
#[link_section="__TEXT,__moretext"]
fn i_live_in_more_text() -> &'static str {
    "knock knock"
}

#[cfg(target_os = "macos")]
#[link_section="__RODATA,__imm"]
static magic: uint = 42;

#[cfg(target_os = "macos")]
#[link_section="__DATA,__mut"]
static mut frobulator: uint = 0xdeadbeef;

pub fn main() {
    unsafe {
        frobulator = 0xcafebabe;
        println!("{} {} {}", i_live_in_more_text(), magic, frobulator);
    }
}
