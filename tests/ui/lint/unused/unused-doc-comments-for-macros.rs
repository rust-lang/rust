#![deny(unused_doc_comments)]
#![feature(rustc_attrs)]

macro_rules! foo { () => {}; }

fn main() {
    /// line1 //~ ERROR: unused doc comment
    /// line2
    /// line3
    foo!();

    // Ensure we still detect another doc-comment block.
    /// line1 //~ ERROR: unused doc comment
    /// line2
    /// line3
    foo!();

    // Regression test for https://github.com/rust-lang/rust/issues/155701.
    cfg_select! {
        /// line1
        //~^ ERROR: unused doc comment
        debug_assertions => (),
        _ => (),
    }
}
