//@ known-bug: rust-lang/rust#129127
//@ compile-flags: -Zmir-opt-level=5 -Zvalidate-mir -Zcross-crate-inline-threshold=always




pub struct Rows<'a>();

impl<'a> Iterator for Rows<'a> {
    type Item = ();

    fn next() -> Option<Self::Item> {
        let mut rows = Rows();
        rows.map(|row| row).next()
    }
}

fn main() {
    let mut rows = Rows();
    rows.next();
}
