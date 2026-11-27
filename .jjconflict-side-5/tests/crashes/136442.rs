//@ known-bug: #136442
//@ compile-flags: -Copt-level=0 -Zmir-enable-passes=+Inline -Zmir-enable-passes=+JumpThreading --crate-type lib
pub fn problem_thingy(items: &mut impl Iterator<Item = str>) {
    let mut peeker = items.peekable();
    match peeker.peek() {
        Some(_) => (),
        None => return (),
    }
}
