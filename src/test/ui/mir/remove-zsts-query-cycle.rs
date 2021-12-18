// Regression test for #88972. Used to cause a query cycle:
//   optimized mir -> remove zsts -> layout of a generator -> optimized mir.
//
// edition:2018
// compile-flags: --crate-type=lib -Zinline-mir=yes
// build-pass

pub async fn listen() -> Result<(), std::io::Error> {
    let f = do_async();
    std::mem::forget(f);
    Ok(())
}

pub async fn do_async() {
    listen().await.unwrap()
}
