// https://github.com/rust-lang/rust/issues/76501
#![crate_name="foo"]

//@ has 'foo/fn.bloop.html' '//pre' 'pub const fn bloop() -> i32'
/// A useless function that always returns 1.
pub const fn bloop() -> i32 {
    1
}

/// A struct.
pub struct Struct {}

impl Struct {
    //@ has 'foo/struct.Struct.html' '//*[@class="method"]' \
    // 'pub const fn blurp() -> i32'
    /// A useless function that always returns 1.
    pub const fn blurp() -> i32 {
        1
    }
}
