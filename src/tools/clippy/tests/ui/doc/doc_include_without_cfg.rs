#![warn(clippy::doc_include_without_cfg)]
// Should not lint.
#![doc(html_playground_url = "https://playground.example.com/")]
#![doc = include_str!("../approx_const.rs")]
//~^ doc_include_without_cfg
// Should not lint.
#![cfg_attr(feature = "whatever", doc = include_str!("../approx_const.rs"))]
#![cfg_attr(doc, doc = include_str!("../approx_const.rs"))]
#![doc = "some doc"]
//! more doc

macro_rules! man_link {
    ($a:literal, $b:literal) => {
        concat!($a, $b)
    };
}

// Should not lint!
macro_rules! tst {
    ($(#[$attr:meta])*) => {
        $(#[$attr])*
        fn blue() {
            println!("Hello, world!");
        }
    }
}

tst! {
    /// This is a test with no included file
}

#[doc = include_str!("../approx_const.rs")]
//~^ doc_include_without_cfg
// Should not lint.
#[doc = man_link!("bla", "blob")]
#[cfg_attr(feature = "whatever", doc = include_str!("../approx_const.rs"))]
#[cfg_attr(doc, doc = include_str!("../approx_const.rs"))]
#[doc = "some doc"]
/// more doc
fn main() {
    // test code goes here
}
