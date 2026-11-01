//! Regression test for issue https://github.com/rust-lang/rust/issues/71798
// ICE with escaping bound variables when impl Future + '_
// returns non-Future type combined with syntax errors

fn test_ref(x: &u32) -> impl std::future::Future<Output = u32> + '_ {
    //~^ ERROR `u32` is not a future
    *x
}

fn main() {
    let _ = test_ref & u; //~ ERROR cannot find value `u` in this scope
}
