use super::{Header, MIN_ALIGN};

#[test]
fn alloc_header() {
    // Header must fit in the padding before an aligned pointer
    assert!(size_of::<Header>() <= MIN_ALIGN);
    assert!(align_of::<Header>() <= MIN_ALIGN);
}
