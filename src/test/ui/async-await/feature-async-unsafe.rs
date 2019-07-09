// edition:2018
// gate-test-async_unsafe

struct S {}

impl S {
    #[cfg(FALSE)] async unsafe fn f() {} //~ ERROR async unsafe functions are unstable
}

#[cfg(FALSE)] async unsafe fn g() {} //~ ERROR async unsafe functions are unstable

fn main() {}
