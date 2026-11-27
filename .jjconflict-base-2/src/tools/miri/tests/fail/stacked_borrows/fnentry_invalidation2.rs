// Regression test for https://github.com/rust-lang/miri/issues/2536
// This tests that we don't try to back too far up the stack when selecting a span to report.
// We should display the as_mut_ptr() call as the location of the invalidation, not the call to
// inner

struct Thing<'a> {
    sli: &'a mut [i32],
}

fn main() {
    let mut t = Thing { sli: &mut [0, 1, 2] };
    let ptr = t.sli.as_ptr();
    inner(&mut t);
    unsafe {
        let _oof = *ptr; //~ ERROR: /read access .* tag does not exist in the borrow stack/
    }
}

fn inner(t: &mut Thing) {
    let _ = t.sli.as_mut_ptr();
}
