use super::{Header, MIN_ALIGN};
use crate::mem;

#[test]
fn alloc_header() {
    // Header must fit in the padding before an aligned pointer
    assert!(mem::size_of::<Header>() <= MIN_ALIGN);
    assert!(mem::align_of::<Header>() <= MIN_ALIGN);
}
