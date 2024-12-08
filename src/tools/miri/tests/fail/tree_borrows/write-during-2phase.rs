//@compile-flags: -Zmiri-tree-borrows

// We invalidate a reference during a 2-phase borrow by doing a Foreign
// Write in between the initial reborrow and function entry. UB occurs
// on function entry when reborrow from a Disabled fails.
// This test would pass under Stacked Borrows, but Tree Borrows
// is more strict on 2-phase borrows.

struct Foo(u64);
impl Foo {
    fn add(&mut self, n: u64) -> u64 {
        //~^ ERROR: /reborrow through .* is forbidden/
        self.0 + n
    }
}

pub fn main() {
    let mut f = Foo(0);
    let alias = &mut f.0 as *mut u64;
    let res = f.add(unsafe {
        // This is the access at fault, but it's not immediately apparent because
        // the reference that got invalidated is not under a Protector.
        *alias = 42;
        0
    });
    // `res` could be optimized to be `0`, since at the time the reference for the `self` argument
    // is created, it has value `0`, and then later we add `0` to that. But turns out there is
    // a sneaky alias that's used to change the value of `*self` before it is read...
    assert_eq!(res, 42);
}
