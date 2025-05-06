// This test serves to document the change in semantics introduced by #138961.
//
// A corollary of partial-pattern.rs: while the tuple access showcases the
// utility, it is actually the dereference performed by the pattern that
// matters.

fn main() {
    // the inner reference is dangling
    let x: &&u32 = unsafe {
        let x: u32 = 42;
        &&* &raw const x
    };

    let _ = || { //~ ERROR: encountered a dangling reference
        match x {
            &&_y => {},
        }
    };
}
