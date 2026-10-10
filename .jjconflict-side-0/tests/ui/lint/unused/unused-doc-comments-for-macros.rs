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

    // Even invalid doc attributes should emit the warning.
    #[doc = {
        let a = 1;
        let b = 1;
        let sum = a + b;
        assert_eq!(sum, 2);
    }]
    //~^^^^^^ ERROR: unused doc comment
    foo!();
}
