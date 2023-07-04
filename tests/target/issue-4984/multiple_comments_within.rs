#[derive(
    /* ---------- Some really important comment that just had to go inside the derive --------- */
    Debug,
    Clone,
    /* Another comment */ Eq,
    PartialEq,
)]
struct Foo {
    a: i32,
    b: T,
}
