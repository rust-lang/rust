// Make sure we don't suggest remove redundant semicolon inside macro expansion.(issue #142143)

#![deny(redundant_semicolons)]

macro_rules! m {
    ($stmt:stmt) => { #[allow(bad_style)] $stmt }
}

fn main() {
    m!(;); //~ ERROR unnecessary trailing semicolon [redundant_semicolons]
}
