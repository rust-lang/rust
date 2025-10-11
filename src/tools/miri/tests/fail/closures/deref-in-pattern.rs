// This test serves to document the change in semantics introduced by
// rust-lang/rust#138961.
//
// A corollary of partial-pattern.rs: while the tuple access testcase makes
// it clear why these semantics are useful, it is actually the dereference
// being performed by the pattern that matters.

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
