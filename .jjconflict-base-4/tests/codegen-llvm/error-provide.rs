// Codegen test for #126242

//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]
#![feature(error_generic_member_access)]
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
    // CHECK-LABEL: @provide
    #[no_mangle]
    fn provide<'a>(&'a self, request: &mut Request<'a>) {
        // LLVM should be able to optimize multiple .provide_* calls into a switch table
        // and eliminate redundant ones, rather than compare one-by-one.

        // CHECK-NEXT: start:
        // CHECK-NEXT: %[[SCRUTINEE:[^ ]+]] = load i128, ptr
        // CHECK-NEXT: switch i128 %[[SCRUTINEE]], label %{{.*}} [
        // CHECK-COUNT-3: i128 {{.*}}, label %{{.*}}
        // CHECK-NEXT: ]
        request
            .provide_ref::<MyBacktrace1>(&self.backtrace1)
            .provide_ref::<MyBacktrace3>(&self.other)
            .provide_ref::<MyBacktrace2>(&self.backtrace2)
            .provide_ref::<MyBacktrace3>(&self.backtrace3);
    }
}
