// @has 'issue_76501/fn.bloop.html' '//pre' 'pub const fn bloop() -> i32'
/// A useless function that always returns 1.
pub const fn bloop() -> i32 {
    1
}

/// A struct.
pub struct Struct {}

impl Struct {
    // @has 'issue_76501/struct.Struct.html' '//*[@class="method"]' \
    // 'pub const fn blurp() -> i32'
    /// A useless function that always returns 1.
    pub const fn blurp() -> i32 {
        1
    }
}
