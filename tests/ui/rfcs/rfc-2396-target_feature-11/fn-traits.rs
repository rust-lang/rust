//@ only-x86_64

#[target_feature(enable = "avx")]
fn foo() {}

#[target_feature(enable = "avx")]
fn bar(arg: i32) {}

#[target_feature(enable = "avx")]
unsafe fn foo_unsafe() {}

fn call(f: impl Fn()) {
    f()
}

fn call_mut(mut f: impl FnMut()) {
    f()
}

fn call_once(f: impl FnOnce()) {
    f()
}

fn call_once_i32(f: impl FnOnce(i32)) {
    f(0)
}

fn main() {
    call(foo); //~ ERROR expected `Fn()` signature, found `#[target_feature(..)] fn() {foo}`
    call_mut(foo); //~ ERROR expected `FnMut()` signature, found `#[target_feature(..)] fn() {foo}`
    call_once(foo); //~ ERROR expected `FnOnce()` signature, found `#[target_feature(..)] fn() {foo}`
    call_once_i32(bar); //~ ERROR expected `FnOnce(i32)` signature, found `#[target_feature(..)] fn(i32) {bar}`

    call(foo_unsafe);
    //~^ ERROR expected `Fn()` signature, found `unsafe fn() {foo_unsafe}`
    call_mut(foo_unsafe);
    //~^ ERROR expected `FnMut()` signature, found `unsafe fn() {foo_unsafe}`
    call_once(foo_unsafe);
    //~^ ERROR expected `FnOnce()` signature, found `unsafe fn() {foo_unsafe}`
}
