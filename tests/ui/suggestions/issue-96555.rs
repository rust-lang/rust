//@ edition:2018

async fn f() {
    m::f1().await; //~ ERROR `()` is not a future
    m::f2().await; //~ ERROR `()` is not a future
    m::f3().await; //~ ERROR `()` is not a future
}

mod m {
    pub fn f1() {}

    pub(crate) fn f2() {}

    pub
    fn
    f3() {}
}

fn main() {}
