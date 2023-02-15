// run-rustfix
// edition:2021
#![allow(dead_code, unused_mut, unused_variables)]
struct A {}
struct Msg<'a> {
    s: &'a [i32],
}
impl A {
    async fn g(buf: &[i32]) -> Msg<'_> {
        Msg { s: &buf[0..1] }
    }
    async fn f() {
        let mut buf = [0; 512];
        let m2 = &buf[..]; //~ ERROR `buf` does not live long enough
        let m = Self::g(m2).await;
        Self::f2(m).await;
    }
    async fn f2(m: Msg) {}
    //~^ ERROR implicit elided lifetime not allowed here
}

fn main() {}
