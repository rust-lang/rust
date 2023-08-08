// revisions: lib staticlib
// ignore-emscripten default visibility is hidden
// compile-flags: -O
// [lib] compile-flags: --crate-type lib
// [staticlib] compile-flags: --crate-type staticlib
// `#[no_mangle]`d static variables always have external linkage, i.e., no `internal` in their
// definitions

// CHECK: @A = {{(dso_local )?}}local_unnamed_addr constant
#[no_mangle]
static A: u8 = 0;

// CHECK: @B = {{(dso_local )?}}local_unnamed_addr global
#[no_mangle]
static mut B: u8 = 0;

// CHECK: @C = {{(dso_local )?}}local_unnamed_addr constant
#[no_mangle]
pub static C: u8 = 0;

// CHECK: @D = {{(dso_local )?}}local_unnamed_addr global
#[no_mangle]
pub static mut D: u8 = 0;

mod private {
    // CHECK: @E = {{(dso_local )?}}local_unnamed_addr constant
    #[no_mangle]
    static E: u8 = 0;

    // CHECK: @F = {{(dso_local )?}}local_unnamed_addr global
    #[no_mangle]
    static mut F: u8 = 0;

    // CHECK: @G = {{(dso_local )?}}local_unnamed_addr constant
    #[no_mangle]
    pub static G: u8 = 0;

    // CHECK: @H = {{(dso_local )?}}local_unnamed_addr global
    #[no_mangle]
    pub static mut H: u8 = 0;
}

const HIDDEN: () = {
    // CHECK: @I = {{(dso_local )?}}local_unnamed_addr constant
    #[no_mangle]
    static I: u8 = 0;

    // CHECK: @J = {{(dso_local )?}}local_unnamed_addr global
    #[no_mangle]
    static mut J: u8 = 0;

    // CHECK: @K = {{(dso_local )?}}local_unnamed_addr constant
    #[no_mangle]
    pub static K: u8 = 0;

    // CHECK: @L = {{(dso_local )?}}local_unnamed_addr global
    #[no_mangle]
    pub static mut L: u8 = 0;
};

fn x() {
    // CHECK: @M = {{(dso_local )?}}local_unnamed_addr constant
    #[no_mangle]
    static M: fn() = x;

    // CHECK: @N = {{(dso_local )?}}local_unnamed_addr global
    #[no_mangle]
    static mut N: u8 = 0;

    // CHECK: @O = {{(dso_local )?}}local_unnamed_addr constant
    #[no_mangle]
    pub static O: u8 = 0;

    // CHECK: @P = {{(dso_local )?}}local_unnamed_addr global
    #[no_mangle]
    pub static mut P: u8 = 0;
}
