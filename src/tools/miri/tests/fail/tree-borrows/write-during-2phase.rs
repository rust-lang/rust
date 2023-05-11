//@compile-flags: -Zmiri-tree-borrows

// We invalidate a reference during a 2-phase borrow by doing a Foreign
// Write in between the initial reborrow and function entry. UB occurs
// on function entry when reborrow from a Disabled fails.
// This test would pass under Stacked Borrows, but Tree Borrows
// is more strict on 2-phase borrows.

struct Foo(u64);
impl Foo {
    #[rustfmt::skip] // rustfmt is wrong about which line contains an error
    fn add(&mut self, n: u64) -> u64 { //~ ERROR: /read access through .* is forbidden/
        self.0 + n
    }
}

pub fn main() {
    let mut f = Foo(0);
    let inner = &mut f.0 as *mut u64;
    let _res = f.add(unsafe {
        let n = f.0;
        // This is the access at fault, but it's not immediately apparent because
        // the reference that got invalidated is not under a Protector.
        *inner = 42;
        n
    });
}
