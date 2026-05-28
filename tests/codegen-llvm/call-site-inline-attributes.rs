//@ compile-flags: -O -Zinline-mir=no -Cno-prepopulate-passes -Zmerge-functions=disabled

#![crate_type = "lib"]

// This test checks that we add inlinehint for #[inline], noinline for #[inline(never)], and
// alwaysinline for #[inline(always)] to call sites.

#[unsafe(no_mangle)]
fn calls_something_noinline() {
    // CHECK-LABEL @calls_something_noinline
    // CHECK: call void @{{.*}}noinline_fn() #[[NOINLINE:[0-9]+]]
    noinline_fn();
}

#[inline(never)]
fn noinline_fn() {}

#[unsafe(no_mangle)]
fn calls_something_inline() {
    // CHECK-LABEL @calls_something_inlinehint
    // CHECK: call void @{{.*}}inlinehint_fn() #[[INLINEHINT:[0-9]+]]
    inlinehint_fn();
}

#[inline]
fn inlinehint_fn() {}

#[unsafe(no_mangle)]
fn calls_something_alwaysinline() {
    // CHECK-LABEL @calls_something_alwaysinline
    // CHECK: call void @{{.*}}alwaysinline_fn() #[[ALWAYSINLINE:[0-9]+]]
    alwaysinline_fn();
}

#[inline(always)]
fn alwaysinline_fn() {}

//CHECK: attributes #[[NOINLINE]] = {{{.*}} noinline {{.*}}}
//CHECK: attributes #[[INLINEHINT]] = {{{.*}} inlinehint {{.*}}}
//CHECK: attributes #[[ALWAYSINLINE]] = {{{.*}} alwaysinline {{.*}}}
