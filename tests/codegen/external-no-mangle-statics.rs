//@ revisions: lib staticlib
//@ ignore-emscripten default visibility is hidden
//@ compile-flags: -Copt-level=3
//@ [lib] compile-flags: --crate-type lib
//@ [staticlib] compile-flags: --crate-type staticlib
// `#[no_mangle]`d static variables always have external linkage, i.e., no `internal` in their
// definitions

// CHECK-DAG: @A = {{(dso_local )?}}local_unnamed_addr constant
#[no_mangle]
static A: u8 = 0;

// CHECK-DAG: @B = {{(dso_local )?}}local_unnamed_addr global
#[no_mangle]
static mut B: u8 = 0;

// CHECK-DAG: @C = {{(dso_local )?}}local_unnamed_addr constant
#[no_mangle]
pub static C: u8 = 0;

// CHECK-DAG: @D = {{(dso_local )?}}local_unnamed_addr global
#[no_mangle]
pub static mut D: u8 = 0;

mod private {
    // CHECK-DAG: @E = {{(dso_local )?}}local_unnamed_addr constant
    #[no_mangle]
    static E: u8 = 0;

    // CHECK-DAG: @F = {{(dso_local )?}}local_unnamed_addr global
    #[no_mangle]
    static mut F: u8 = 0;

    // CHECK-DAG: @G = {{(dso_local )?}}local_unnamed_addr constant
    #[no_mangle]
    pub static G: u8 = 0;

    // CHECK-DAG: @H = {{(dso_local )?}}local_unnamed_addr global
    #[no_mangle]
    pub static mut H: u8 = 0;
}

const HIDDEN: () = {
    // CHECK-DAG: @I = {{(dso_local )?}}local_unnamed_addr constant
    #[no_mangle]
    static I: u8 = 0;

    // CHECK-DAG: @J = {{(dso_local )?}}local_unnamed_addr global
    #[no_mangle]
    static mut J: u8 = 0;

    // CHECK-DAG: @K = {{(dso_local )?}}local_unnamed_addr constant
    #[no_mangle]
    pub static K: u8 = 0;

    // CHECK-DAG: @L = {{(dso_local )?}}local_unnamed_addr global
    #[no_mangle]
    pub static mut L: u8 = 0;
};

fn x() {
    // CHECK-DAG: @M = {{(dso_local )?}}local_unnamed_addr constant
    #[no_mangle]
    static M: fn() = x;

    // CHECK-DAG: @N = {{(dso_local )?}}local_unnamed_addr global
    #[no_mangle]
    static mut N: u8 = 0;

    // CHECK-DAG: @O = {{(dso_local )?}}local_unnamed_addr constant
    #[no_mangle]
    pub static O: u8 = 0;

    // CHECK-DAG: @P = {{(dso_local )?}}local_unnamed_addr global
    #[no_mangle]
    pub static mut P: u8 = 0;
}
