//@ build-pass
//@ compile-flags: -Copt-level=0 -Zmir-enable-passes=+Inline -Zmir-enable-passes=+JumpThreading --crate-type lib

// Regression test for <https://github.com/rust-lang/rust/issues/136442>
// Used to fail in MIR but #138526 (accidentally) worked around that.

pub fn problem_thingy(items: &mut impl Iterator<Item = str>) {
    let mut peeker = items.peekable();
    match peeker.peek() {
        Some(_) => (),
        None => return (),
    }
}
