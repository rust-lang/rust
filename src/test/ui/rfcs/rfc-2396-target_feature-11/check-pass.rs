// Tests the new rules added by RFC 2396, including:
// - applying `#[target_feature]` to safe functions is allowed
// - calling functions with `#[target_feature]` is allowed in
//   functions which have (at least) the same features
// - calling functions with `#[target_feature]` is allowed in
//   unsafe contexts
// - functions with `#[target_feature]` can coerce to unsafe fn pointers

// check-pass
// only-x86_64
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(target_feature_11)]

#[target_feature(enable = "sse2")]
const fn sse2() {}

#[cfg(target_feature = "sse2")]
const SSE2_ONLY: () = unsafe {
    sse2();
};

#[target_feature(enable = "sse2")]
fn also_sse2() {
    sse2();
}

#[target_feature(enable = "sse2")]
#[target_feature(enable = "avx")]
fn sse2_and_avx() {
    sse2();
}

struct Foo;

impl Foo {
    #[target_feature(enable = "sse2")]
    fn sse2(&self) {
        sse2();
    }
}

fn main() {
    if cfg!(target_feature = "sse2") {
        unsafe {
            sse2();
            Foo.sse2();
        }
    }
    let sse2_ptr: unsafe fn() = sse2;
}
