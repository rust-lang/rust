//! Yet again an ICE (#159572) due to impossible clauses that reach MIR optimizations.
//@ compile-flags: -Zmir-enable-passes=+GVN -Zvalidate-mir
//@ build-pass

trait A {
    fn foo(&self) -> Self
    where
        Self: Copy;
}
impl A for [&()] {
    fn foo(&self) -> Self
    where
        Self: Copy,
    {
        *(&[] as &_)
    }
}
fn main() {}
