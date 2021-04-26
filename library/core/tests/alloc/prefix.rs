use core::alloc::helper::PrefixAllocator;
use core::alloc::{Allocator, Layout};
use core::any::type_name;
use std::alloc::System;

use super::Tracker;

fn test_prefix<T, Prefix>() {
    unsafe {
        let layout = Layout::new::<T>();
        let prefix_offset = PrefixAllocator::<System, Prefix>::prefix_offset(layout);
        assert_eq!(
            prefix_offset,
            Layout::new::<Prefix>().extend(layout).unwrap().1,
            "Invalid prefix offset for PrefixAllocator<_, {}> with Layout<{}>.",
            type_name::<Prefix>(),
            type_name::<T>(),
        );

        let alloc =
            Tracker::new(PrefixAllocator::<Tracker<System>, Prefix>::new(Tracker::new(System)));
        let memory = alloc.allocate(layout).unwrap_or_else(|_| {
            panic!(
                "Could not allocate {} bytes for PrefixAllocator<_, {}> with Layout<{}>.",
                layout.size(),
                type_name::<Prefix>(),
                type_name::<T>()
            )
        });

        assert_eq!(
            PrefixAllocator::<System, Prefix>::prefix::<T>(memory.as_non_null_ptr().cast())
                .cast()
                .as_ptr(),
            memory.as_mut_ptr().sub(prefix_offset),
            "Invalid prefix location for PrefixAllocator<_, {}> with Layout<{}>.",
            type_name::<Prefix>(),
            type_name::<T>(),
        );

        alloc.deallocate(memory.as_non_null_ptr(), layout);
    }
}

#[repr(align(1024))]
#[derive(Debug, Copy, Clone, PartialEq)]
struct AlignTo1024<T> {
    a: T,
}

#[repr(align(64))]
#[derive(Debug, Copy, Clone, PartialEq)]
struct AlignTo64;

#[test]
fn test() {
    macro_rules! test_ty {
            ($($ty:ty),*) => { test_ty!(@2 $($ty),*; ($($ty),*)) };
            (@2 $($tyl:ty),*; $tyr:tt) => { $(test_ty!(@3 $tyl; $tyr);)* };
            (@3 $tyl:ty; ($($tyr:ty),*)) => { $(test_prefix::<$tyl, $tyr>();)* };
        }
    // call test_pair::<A, B>() for every combination of these types
    test_ty!(
        (),
        u8,
        u16,
        u32,
        u64,
        u128,
        AlignTo64,
        AlignTo1024<u8>,
        AlignTo1024<u16>,
        AlignTo1024<u32>
    );
}
