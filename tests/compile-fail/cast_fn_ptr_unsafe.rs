// just making sure that fn -> unsafe fn casts are handled by rustc so miri doesn't have to
fn main() {
    fn f() {}

    let g = f as fn() as unsafe fn(i32); //~ERROR: non-primitive cast: `fn()` as `unsafe fn(i32)`

    unsafe {
        g(42);
    }
}
