use super::*;

extern crate test;
use crate::boxed::Box;
use test::Bencher;

#[test]
fn allocate_zeroed() {
    unsafe {
        let layout = Layout::from_size_align(1024, 1).unwrap();
        let ptr =
            Global.alloc_zeroed(layout.clone()).unwrap_or_else(|_| handle_alloc_error(layout));

        let mut i = ptr.cast::<u8>().as_ptr();
        let end = i.add(layout.size());
        while i < end {
            assert_eq!(*i, 0);
            i = i.offset(1);
        }
        Global.dealloc(ptr, layout);
    }
}

#[bench]
#[cfg_attr(miri, ignore)] // Miri does not support benchmarks
fn alloc_owned_small(b: &mut Bencher) {
    b.iter(|| {
        let _: Box<_> = box 10;
    })
}
