//@ edition: 2024

#![feature(gpu_offload)]
#![feature(offload)]

use core::offload::PartitioningStrategy;

#[derive(Debug, Clone, Copy)]
pub struct Dummy;

unsafe impl PartitioningStrategy for Dummy {
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    fn index() -> usize {
        0
    }

    unsafe fn get<'a, T>(_ptr: *const T, _len: usize) -> Option<Self::View<'a, T>> {
        None
    }

    unsafe fn get_mut<'a, T>(_ptr: *mut T, _len: usize) -> Option<Self::ViewMut<'a, T>> {
        None
    }
}
