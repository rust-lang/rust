//! Test for https://github.com/rust-lang/rust/issues/64219
//! Check if `noreturn` attribute is applied on calls to
//! function pointers returning `!` (never type).

#![crate_type = "lib"]

extern "C" {
    static FOO: fn() -> !;
}

// CHECK-LABEL: @foo
#[no_mangle]
pub unsafe fn foo() {
    // CHECK: call
    // CHECK-SAME: [[NUM:#[0-9]+$]]
    FOO();
}

// CHECK: attributes [[NUM]] = {{{.*}} noreturn {{.*}}}
