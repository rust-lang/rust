// https://github.com/rust-lang/rust/pull/83872#issuecomment-820101008
#![crate_name="foo"]

mod sub4 {
    /// 0
    pub const X: usize = 0;
    pub mod inner {
        pub use super::*;
        /// 1
        pub const X: usize = 1;
    }
}

#[doc(inline)]
pub use sub4::inner::*;

// @has 'foo/index.html'
// @has - '//div[@class="item-right docblock-short"]' '1'
// @!has - '//div[@class="item-right docblock-short"]' '0'
fn main() { assert_eq!(X, 1); }
