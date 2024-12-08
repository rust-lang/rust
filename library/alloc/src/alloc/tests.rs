use super::*;

extern crate test;
use test::Bencher;

use crate::boxed::Box;

#[test]
fn allocate_zeroed() {
    unsafe {
        let layout = Layout::from_size_align(1024, 1).unwrap();
        let ptr =
            Global.allocate_zeroed(layout.clone()).unwrap_or_else(|_| handle_alloc_error(layout));

        let mut i = ptr.as_non_null_ptr().as_ptr();
        let end = i.add(layout.size());
        while i < end {
            assert_eq!(*i, 0);
            i = i.add(1);
        }
        Global.deallocate(ptr.as_non_null_ptr(), layout);
    }
}

#[bench]
fn alloc_owned_small(b: &mut Bencher) {
    b.iter(|| {
        let _: Box<_> = Box::new(10);
    })
}
