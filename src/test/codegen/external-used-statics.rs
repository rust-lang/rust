// revisions: lib staticlib
// ignore-emscripten default visibility is hidden
// compile-flags: -O
// `#[used]`d static variables always have external linkage, i.e., no `internal` in their
// definitions

#![cfg_attr(lib, crate_type = "lib")]
#![cfg_attr(staticlib, crate_type = "staticlib")]

// CHECK: @{{.*}}A{{.*}} = constant
#[used]
static A: u8 = 0;

// CHECK: @{{.*}}B{{.*}} = global
#[used]
static mut B: u8 = 0;

// CHECK: @{{.*}}C{{.*}} = constant
#[used]
pub static C: u8 = 0;

// CHECK: @{{.*}}D{{.*}} = global
#[used]
pub static mut D: u8 = 0;

mod private {
    // CHECK: @{{.*}}E{{.*}} = constant
    #[used]
    static E: u8 = 0;

    // CHECK: @{{.*}}F{{.*}} = global
    #[used]
    static mut F: u8 = 0;

    // CHECK: @{{.*}}G{{.*}} = constant
    #[used]
    pub static G: u8 = 0;

    // CHECK: @{{.*}}H{{.*}} = global
    #[used]
    pub static mut H: u8 = 0;
}

const HIDDEN: () = {
    // CHECK: @{{.*}}I{{.*}} = constant
    #[used]
    static I: u8 = 0;

    // CHECK: @{{.*}}J{{.*}} = global
    #[used]
    static mut J: u8 = 0;

    // CHECK: @{{.*}}K{{.*}} = constant
    #[used]
    pub static K: u8 = 0;

    // CHECK: @{{.*}}L{{.*}} = global
    #[used]
    pub static mut L: u8 = 0;
};

fn x() {
    // CHECK: @{{.*}}M{{.*}} = constant
    #[used]
    static M: fn() = x;

    // CHECK: @{{.*}}N{{.*}} = global
    #[used]
    static mut N: u8 = 0;

    // CHECK: @{{.*}}O{{.*}} = constant
    #[used]
    pub static O: u8 = 0;

    // CHECK: @{{.*}}P{{.*}} = global
    #[used]
    pub static mut P: u8 = 0;
}
