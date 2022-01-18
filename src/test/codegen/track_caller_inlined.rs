// This test makes sure that calls to std::panic::Location::caller()
// don't result in an actual function call. The caller location is
// known at compile time so the call can always be optimized away.

// compile-flags: -Copt-level=2

#![crate_type = "lib"]
#![feature(bench_black_box)]

// The first check makes sure that the caller location is used at all,
// i.e. that std::hint::black_box() works.
// CHECK: %0 = alloca %"core::panic::location::Location"*

// This check makes sure that no call to `std::panic::Location::caller()`
// is emitted. The sequence of characters is valid for both v0 and legacy
// mangling.
// CHECK-NOT: call {{.*}}8Location6caller

// CHECK: call void asm sideeffect {{.*}}(%"core::panic::location::Location"** nonnull %0)

#[track_caller]
fn foo() {
    std::hint::black_box(std::panic::Location::caller());
}

pub fn bar() {
    foo();
}
