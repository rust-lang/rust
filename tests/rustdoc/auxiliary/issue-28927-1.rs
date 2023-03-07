mod detail {
    pub extern crate issue_28927_2 as inner2;
}
pub use detail::inner2 as bar;
