// compile-flags: -O -C debuginfo=0 -Zmir-opt-level=2
// only-64bit
// ignore-debug

#![crate_type = "lib"]

// When this test was added, the MIR for `next` was 174 lines just for the basic
// blocks -- far more if you counted the scopes.  The goal of having this here
// is to hopefully keep it a reasonable size, ideally eventually small enough
// that the mir inliner would actually be willing to inline it, since it's an
// important building block and usually very few *backend* instructions.

// As such, feel free to `--bless` whatever changes you get here, so long as
// doing so doesn't add substantially more MIR.

// EMIT_MIR slice_iter.slice_iter_next.PreCodegen.after.mir
pub fn slice_iter_next<'a, T>(it: &mut std::slice::Iter<'a, T>) -> Option<&'a T> {
    it.next()
}

// EMIT_MIR slice_iter.slice_iter_mut_next_back.PreCodegen.after.mir
pub fn slice_iter_mut_next_back<'a, T>(it: &mut std::slice::IterMut<'a, T>) -> Option<&'a mut T> {
    it.next_back()
}

// EMIT_MIR slice_iter.forward_loop.PreCodegen.after.mir
pub fn forward_loop<'a, T>(slice: &'a [T], f: impl Fn(&T)) {
    for x in slice.iter() {
        f(x)
    }
}

// EMIT_MIR slice_iter.reverse_loop.PreCodegen.after.mir
pub fn reverse_loop<'a, T>(slice: &'a [T], f: impl Fn(&T)) {
    for x in slice.iter().rev() {
        f(x)
    }
}

// EMIT_MIR slice_iter.enumerated_loop.PreCodegen.after.mir
pub fn enumerated_loop<'a, T>(slice: &'a [T], f: impl Fn(usize, &T)) {
    for (i, x) in slice.iter().enumerate() {
        f(i, x)
    }
}

// EMIT_MIR slice_iter.range_loop.PreCodegen.after.mir
pub fn range_loop<'a, T>(slice: &'a [T], f: impl Fn(usize, &T)) {
    for i in 0..slice.len() {
        let x = &slice[i];
        f(i, x)
    }
}
