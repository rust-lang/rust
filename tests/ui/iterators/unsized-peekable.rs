//@ compile-flags: -Zmir-enable-passes=+Inline -Zmir-enable-passes=+JumpThreading --crate-type=lib

pub fn problem_thingy(items: &mut impl Iterator<Item = str>) {
    let mut peeker = items.peekable();
    //~^ ERROR: the size for values of type `str` cannot be known at compilation time [E0277]
    // ^^^ doesn't have a size known at compile-time
    match peeker.peek() {
        Some(_) => (),
        None => return (),
    }
}
