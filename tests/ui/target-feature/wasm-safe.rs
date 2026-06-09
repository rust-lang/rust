//@ only-wasm32
//@ check-pass

#![feature(wasm_target_feature)]
#![allow(dead_code)]

#[target_feature(enable = "nontrapping-fptoint")]
fn foo() {}

#[target_feature(enable = "nontrapping-fptoint")]
extern "C" fn bar() {}

trait A {
    fn foo();
    fn bar(&self);
}

struct B;

impl B {
    #[target_feature(enable = "nontrapping-fptoint")]
    fn foo() {}
    #[target_feature(enable = "nontrapping-fptoint")]
    fn bar(&self) {}
}

impl A for B {
    #[target_feature(enable = "nontrapping-fptoint")]
    fn foo() {}
    #[target_feature(enable = "nontrapping-fptoint")]
    fn bar(&self) {}
}

fn no_features_enabled_on_this_function() {
    bar();
    foo();
    B.bar();
    B::foo();
    <B as A>::foo();
    <B as A>::bar(&B);
}

#[target_feature(enable = "nontrapping-fptoint")]
fn main() {}
