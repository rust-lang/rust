// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![feature(const_fn)]

const bad : u32 = {
    {
        5;
        //~^ ERROR statements in constants are unstable
        0
    }
};

const bad_two : u32 = {
    {
        invalid();
        //~^ ERROR statements in constants are unstable
        //~^^ ERROR: calls in constants are limited to constant functions, tuple structs and tuple variants
        0
    }
};

const bad_three : u32 = {
    {
        valid();
        //~^ ERROR statements in constants are unstable
        0
    }
};

static bad_four : u32 = {
    {
        5;
        //~^ ERROR statements in statics are unstable
        0
    }
};

static bad_five : u32 = {
    {
        invalid();
        //~^ ERROR: calls in statics are limited to constant functions, tuple structs and tuple variants
        //~| ERROR statements in statics are unstable
        0
    }
};

static bad_six : u32 = {
    {
        valid();
        //~^ ERROR statements in statics are unstable
        0
    }
};

static mut bad_seven : u32 = {
    {
        5;
        //~^ ERROR statements in statics are unstable
        0
    }
};

static mut bad_eight : u32 = {
    {
        invalid();
        //~^ ERROR statements in statics are unstable
        //~| ERROR: calls in statics are limited to constant functions, tuple structs and tuple variants
        0
    }
};

static mut bad_nine : u32 = {
    {
        valid();
        //~^ ERROR statements in statics are unstable
        0
    }
};


fn invalid() {}
const fn valid() {}

fn main() {}
