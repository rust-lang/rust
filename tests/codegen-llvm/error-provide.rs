// Codegen test for #126242

//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]
#![feature(error_generic_member_access, error_generic_member_multi_access)]

extern crate core;

use core::error::provide::MultiResponse;
use std::error::Request;
use std::fmt;

#[derive(Debug)]
struct MyBacktrace1 {}

#[derive(Debug)]
struct MyBacktrace2 {}

#[derive(Debug)]
struct MyBacktrace3 {}

#[derive(Debug)]
struct MyError {
    backtrace1: MyBacktrace1,
    backtrace2: MyBacktrace2,
    backtrace3: MyBacktrace3,
    other: MyBacktrace3,
}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Example Error")
    }
}

impl std::error::Error for MyError {
    #[no_mangle]
    fn provide<'a>(&'a self, request: &mut Request<'a>) {
        // In an ideal world, LLVM would be able to generate a jump table here.
        // Currently it can not, mostly because it can't prove that the tag id
        // is not modified. However, this shouldn't matter much since this
        // API shouldn't be called many times - when requesting a large number
        // of items, MultiRequestBuilder should be used.
        //
        // We could make a MIR optimization pass that flattens
        // the reads of the tag id - this is sound, since it is mutably borrowed,
        // but this has a fairly low cost/benefit ratio - `provide` should
        // only be called O(1) times per error constructed, and it's already
        // not much slower than constructing the error (and much faster
        // if an allocation, backtrace or formatting is involved).
        request
            .provide_ref::<MyBacktrace1>(&self.backtrace1)
            .provide_ref::<MyBacktrace3>(&self.other)
            .provide_ref::<MyBacktrace2>(&self.backtrace2)
            .provide_ref::<MyBacktrace3>(&self.backtrace3);
    }
}

pub fn provide_multi(
    e: &dyn std::error::Error,
) -> (Option<&[u8; 0]>, Option<&[u8; 1]>, Option<&[u8; 2]>) {
    let mut request = core::error::provide::MultiRequestBuilder::new()
        .with_ref::<[u8; 0]>()
        .with_ref::<[u8; 1]>()
        .with_ref::<[u8; 2]>()
        .with_ref::<[u8; 3]>()
        .with_ref::<[u8; 4]>()
        .with_ref::<[u8; 5]>()
        .with_ref::<[u8; 6]>()
        .with_ref::<[u8; 7]>()
        .request(e);
    (request.retrieve_ref(), request.retrieve_ref(), request.retrieve_ref())
}

// Check that the virtual function generated has a switch

// CHECK: define {{.*}}4core5error7provide{{.*}}21ChainRefMultiResponse
// CHECK-NEXT: start:
// CHECK-NEXT: %[[SCRUTINEE:[^ ]+]] = load i128, ptr
// CHECK-NEXT: switch i128 %[[SCRUTINEE]], label %{{.*}} [
// The request we write has 8 arms. However, LLVM puts the "bottom-most" 3 arms in
// a separate branch, before it produces a switch. This still leaves us with
// O(log n) performance [LLVM generates a binary tree for ]
// CHECK-COUNT-5: i128 {{.*}}, label %{{.*}}
// CHECK-NEXT: ]
