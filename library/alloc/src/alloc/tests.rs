use super::*;

extern crate test;
use crate::boxed::Box;
use test::Bencher;

#[test]
fn allocate_zeroed() {
    unsafe {
        let layout = Layout::from_size_align(1024, 1).unwrap();
        let memory = Global
            .alloc(layout.clone(), AllocInit::Zeroed)
            .unwrap_or_else(|_| handle_alloc_error(layout));

        let mut i = memory.ptr.cast::<u8>().as_ptr();
        let end = i.add(layout.size());
        while i < end {
            assert_eq!(*i, 0);
            i = i.offset(1);
        }
        Global.dealloc(memory.ptr, layout);
    }
}

#[bench]
#[cfg_attr(miri, ignore)] // isolated Miri does not support benchmarks
fn alloc_owned_small(b: &mut Bencher) {
    b.iter(|| {
        let _: Box<_> = box 10;
    })
}
