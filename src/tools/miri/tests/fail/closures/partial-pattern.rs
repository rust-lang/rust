// This test serves to document the change in semantics introduced by
// rust-lang/rust#138961.
//
// Previously, the closure would capture the entirety of x, and access *(*x).0
// when called. Now, the closure only captures *(*x).0, which means that
// a &*(*x).0 reborrow happens when the closure is constructed.
//
// Hence, if one of the references is dangling, this constitutes newly introduced UB
// in the case where the closure doesn't get called. This isn't a big deal,
// because while opsem only now considers this to be UB, the unsafe code
// guidelines have long recommended against any handling of dangling references.

fn main() {
    // the inner references are dangling
    let x: &(&u32, &u32) = unsafe {
        let a = 21;
        let b = 37;
        let ra = &*&raw const a;
        let rb = &*&raw const b;
        &(ra, rb)
    };

    //~v ERROR: encountered a dangling reference
    let _ = || {
        match x {
            (&_y, _) => {}
        }
    };
}
