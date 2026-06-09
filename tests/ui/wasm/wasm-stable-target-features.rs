//@ only-wasm32
//@ build-pass

// Test that a variety of WebAssembly features are all stable and can be used in
// `#[target_feature]`. That should mean they're also available via
// `#[cfg(target_feature)]` as well.

#[target_feature(enable = "multivalue")]
fn foo1() {}

#[target_feature(enable = "reference-types")]
fn foo2() {}

#[target_feature(enable = "bulk-memory")]
fn foo3() {}

#[target_feature(enable = "extended-const")]
fn foo4() {}

#[target_feature(enable = "mutable-globals")]
fn foo5() {}

#[target_feature(enable = "nontrapping-fptoint")]
fn foo6() {}

#[target_feature(enable = "simd128")]
fn foo7() {}

#[target_feature(enable = "relaxed-simd")]
fn foo8() {}

#[target_feature(enable = "sign-ext")]
fn foo9() {}

#[target_feature(enable = "tail-call")]
fn foo10() {}

fn main() {
    foo1();
    foo2();
    foo3();
    foo4();
    foo5();
    foo6();
    foo7();
    foo8();
    foo9();
    foo10();
}
