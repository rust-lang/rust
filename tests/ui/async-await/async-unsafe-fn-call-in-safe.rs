//@ edition:2018

struct S;

impl S {
    async unsafe fn f() {}
}

async unsafe fn f() {}

async fn g() {
    S::f();
    //~^ ERROR call to unsafe function `S::f` is unsafe
    f();
    //~^ ERROR call to unsafe function `f` is unsafe
}

fn main() {
    S::f();
    //~^ ERROR call to unsafe function `S::f` is unsafe
    f();
    //~^ ERROR call to unsafe function `f` is unsafe
}
