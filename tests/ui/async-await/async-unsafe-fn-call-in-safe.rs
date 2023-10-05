// edition:2018
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

struct S;

impl S {
    async unsafe fn f() {}
}

async unsafe fn f() {}

async fn g() {
    S::f();
    //[mir]~^ ERROR call to unsafe function is unsafe
    //[thir]~^^ ERROR call to unsafe function `S::f` is unsafe
    f();
    //[mir]~^ ERROR call to unsafe function is unsafe
    //[thir]~^^ ERROR call to unsafe function `f` is unsafe
}

fn main() {
    S::f();
    //[mir]~^ ERROR call to unsafe function is unsafe
    //[thir]~^^ ERROR call to unsafe function `S::f` is unsafe
    f();
    //[mir]~^ ERROR call to unsafe function is unsafe
    //[thir]~^^ ERROR call to unsafe function `f` is unsafe
}
