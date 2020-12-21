use super::StructAlloc;
use crate::alloc::Global;
use crate::collections::VecDeque;
use core::alloc::{AllocError, Allocator};
use std::alloc::Layout;
use std::cell::RefCell;
use std::ptr::NonNull;
use std::{any, ptr};

fn test_pair<T, Data>() {
    if let Err(_) = std::panic::catch_unwind(|| test_pair_::<T, Data>()) {
        panic!("test of {} followed by {} failed", any::type_name::<T>(), any::type_name::<Data>());
    }
}

fn test_pair_<T, Data>() {
    #[repr(C)]
    struct S<T, Data> {
        t: T,
        data: Data,
    }

    let offset = {
        let s: *const S<T, Data> = ptr::null();
        unsafe { std::ptr::addr_of!((*s).data) as usize }
    };

    let expected_layout = RefCell::new(VecDeque::new());
    let expected_ptr = RefCell::new(VecDeque::new());

    let check_layout = |actual| {
        let mut e = expected_layout.borrow_mut();
        match e.pop_front() {
            Some(expected) if expected == actual => {}
            Some(expected) => panic!("expected layout {:?}, actual layout {:?}", expected, actual),
            _ => panic!("unexpected allocator invocation with layout {:?}", actual),
        }
    };

    let check_ptr = |actual: NonNull<u8>| {
        let mut e = expected_ptr.borrow_mut();
        match e.pop_front() {
            Some(expected) if expected == actual.as_ptr() => {}
            Some(expected) => {
                panic!("expected pointer {:p}, actual pointer {:p}", expected, actual)
            }
            _ => panic!("unexpected allocator invocation with pointer {:p}", actual),
        }
    };

    struct TestAlloc<F, G>(F, G);

    unsafe impl<F, G> Allocator for TestAlloc<F, G>
    where
        F: Fn(Layout),
        G: Fn(NonNull<u8>),
    {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            self.0(layout);
            Global.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            self.1(ptr);
            self.0(layout);
            unsafe { Global.deallocate(ptr, layout) }
        }
    }

    let struct_alloc = StructAlloc::<T, _>::new(TestAlloc(check_layout, check_ptr));

    fn s_layout<T, Data, const N: usize>() -> Layout {
        Layout::new::<S<T, [Data; N]>>()
    }

    fn d_layout<Data, const N: usize>() -> Layout {
        Layout::new::<[Data; N]>()
    }

    fn check_slice<Data, const N: usize>(ptr: NonNull<[u8]>) {
        let expected = d_layout::<Data, N>().size();
        if ptr.len() != expected {
            panic!(
                "expected allocation size: {:?}, actual allocation size: {:?}",
                expected,
                ptr.len()
            )
        }
    }

    expected_layout.borrow_mut().push_back(s_layout::<T, Data, 1>());
    let ptr = struct_alloc.allocate(d_layout::<Data, 1>()).unwrap();
    check_slice::<Data, 1>(ptr);
    unsafe {
        expected_ptr.borrow_mut().push_back(ptr.as_mut_ptr().sub(offset));
    }
    expected_layout.borrow_mut().push_back(s_layout::<T, Data, 3>());
    expected_layout.borrow_mut().push_back(s_layout::<T, Data, 1>());
    let ptr = unsafe {
        struct_alloc
            .grow(ptr.as_non_null_ptr(), d_layout::<Data, 1>(), d_layout::<Data, 3>())
            .unwrap()
    };
    check_slice::<Data, 3>(ptr);
    unsafe {
        expected_ptr.borrow_mut().push_back(ptr.as_mut_ptr().sub(offset));
    }
    expected_layout.borrow_mut().push_back(s_layout::<T, Data, 3>());
    unsafe {
        struct_alloc.deallocate(ptr.as_non_null_ptr(), d_layout::<Data, 3>());
    }
    if !expected_ptr.borrow().is_empty() || !expected_layout.borrow().is_empty() {
        panic!("missing allocator calls");
    }
}

#[test]
fn test() {
    macro_rules! test_ty {
        ($($ty:ty),*) => { test_ty!(@2 $($ty),*; ($($ty),*)) };
        (@2 $($tyl:ty),*; $tyr:tt) => { $(test_ty!(@3 $tyl; $tyr);)* };
        (@3 $tyl:ty; ($($tyr:ty),*)) => { $(test_pair::<$tyl, $tyr>();)* };
    }
    // call test_pair::<A, B>() for every combination of these types
    test_ty!((), u8, u16, u32, u64, u128);
}
