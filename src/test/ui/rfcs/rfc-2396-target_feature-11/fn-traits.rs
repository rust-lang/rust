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
    call(foo); //~ ERROR expected a `Fn<()>` closure, found `[fn item {foo}: fn()]`
    call_mut(foo); //~ ERROR expected a `FnMut<()>` closure, found `[fn item {foo}: fn()]`
    call_once(foo); //~ ERROR expected a `FnOnce<()>` closure, found `[fn item {foo}: fn()]`

    call(foo_unsafe);
    //~^ ERROR expected a `Fn<()>` closure, found `[fn item {foo_unsafe}: unsafe fn()]`
    call_mut(foo_unsafe);
    //~^ ERROR expected a `FnMut<()>` closure, found `[fn item {foo_unsafe}: unsafe fn()]`
    call_once(foo_unsafe);
    //~^ ERROR expected a `FnOnce<()>` closure, found `[fn item {foo_unsafe}: unsafe fn()]`
}
