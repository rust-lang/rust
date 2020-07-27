// only-x86_64

#![feature(target_feature_11)]

#[target_feature(enable = "avx")]
fn foo() {}

#[target_feature(enable = "avx")]
unsafe fn foo_unsafe() {}

fn call(f: impl Fn()) {
    f()
}

fn call_mut(f: impl FnMut()) {
    f()
}

fn call_once(f: impl FnOnce()) {
    f()
}

fn main() {
    call(foo); //~ ERROR expected a `std::ops::Fn<()>` closure, found `fn() {foo}`
    call_mut(foo); //~ ERROR expected a `std::ops::FnMut<()>` closure, found `fn() {foo}`
    call_once(foo); //~ ERROR expected a `std::ops::FnOnce<()>` closure, found `fn() {foo}`

    call(foo_unsafe);
    //~^ ERROR expected a `std::ops::Fn<()>` closure, found `unsafe fn() {foo_unsafe}`
    call_mut(foo_unsafe);
    //~^ ERROR expected a `std::ops::FnMut<()>` closure, found `unsafe fn() {foo_unsafe}`
    call_once(foo_unsafe);
    //~^ ERROR expected a `std::ops::FnOnce<()>` closure, found `unsafe fn() {foo_unsafe}`
}
