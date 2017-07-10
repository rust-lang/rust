// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(warnings)]

#![feature(rustc_private)]

extern crate rustc;
extern crate rustc_errors;
extern crate syntax;
extern crate syntax_pos;

pub mod expand;

pub static ALLOCATOR_METHODS: &[AllocatorMethod] = &[
    AllocatorMethod {
        name: "alloc",
        inputs: &[AllocatorTy::Layout],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: "oom",
        inputs: &[AllocatorTy::AllocErr],
        output: AllocatorTy::Bang,
    },
    AllocatorMethod {
        name: "dealloc",
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout],
        output: AllocatorTy::Unit,
    },
    AllocatorMethod {
        name: "usable_size",
        inputs: &[AllocatorTy::LayoutRef],
        output: AllocatorTy::UsizePair,
    },
    AllocatorMethod {
        name: "realloc",
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout, AllocatorTy::Layout],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: "alloc_zeroed",
        inputs: &[AllocatorTy::Layout],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: "alloc_excess",
        inputs: &[AllocatorTy::Layout],
        output: AllocatorTy::ResultExcess,
    },
    AllocatorMethod {
        name: "realloc_excess",
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout, AllocatorTy::Layout],
        output: AllocatorTy::ResultExcess,
    },
    AllocatorMethod {
        name: "grow_in_place",
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout, AllocatorTy::Layout],
        output: AllocatorTy::ResultUnit,
    },
    AllocatorMethod {
        name: "shrink_in_place",
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout, AllocatorTy::Layout],
        output: AllocatorTy::ResultUnit,
    },
];

pub struct AllocatorMethod {
    pub name: &'static str,
    pub inputs: &'static [AllocatorTy],
    pub output: AllocatorTy,
}

pub enum AllocatorTy {
    AllocErr,
    Bang,
    Layout,
    LayoutRef,
    Ptr,
    ResultExcess,
    ResultPtr,
    ResultUnit,
    Unit,
    UsizePair,
}
