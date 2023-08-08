//! This is a pathological pattern in which opportunities to merge
//! adjacent identical items in the RangeMap are not properly detected
//! because `RangeMap::iter_mut` is never called on overlapping ranges
//! and thus never merges previously split ranges. This does not produce any
//! additional cost for access operations, but it makes the job of the Tree Borrows
//! GC procedure much more costly.
//! See https://github.com/rust-lang/miri/issues/2863

const LENGTH: usize = (1 << 14) - 1;
const LONG: &[u8] = &[b'x'; LENGTH];

fn main() {
    assert!(eq(LONG, LONG))
}

fn eq(s1: &[u8], s2: &[u8]) -> bool {
    if s1.len() != s2.len() {
        return false;
    }

    s1.iter().zip(s2).all(|(c1, c2)| *c1 == *c2)
}
