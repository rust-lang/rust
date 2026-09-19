//@ edition: 2024

#![feature(gpu_offload)]
#![feature(offload)]

use core::offload::{LaunchError, PartitioningStrategy};
use core::ptr::NonNull;

#[derive(Debug, Clone, Copy)]
pub struct Dummy;

unsafe impl PartitioningStrategy for Dummy {
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    fn index() -> usize {
        0
    }

    unsafe fn get<'a, T>(_ptr: NonNull<T>, _len: usize) -> Option<Self::View<'a, T>> {
        None
    }

    unsafe fn get_mut<'a, T>(_ptr: NonNull<T>, _len: usize) -> Option<Self::ViewMut<'a, T>> {
        None
    }

    fn check_launch(_len: usize, _grid: [u32; 3], _block: [u32; 3]) -> Result<(), LaunchError> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Linear1D;

unsafe impl PartitioningStrategy for Linear1D {
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    fn index() -> usize {
        0
    }

    unsafe fn get<'a, T>(_ptr: NonNull<T>, _len: usize) -> Option<Self::View<'a, T>> {
        None
    }

    unsafe fn get_mut<'a, T>(_ptr: NonNull<T>, _len: usize) -> Option<Self::ViewMut<'a, T>> {
        None
    }

    fn check_launch(_len: usize, grid: [u32; 3], block: [u32; 3]) -> Result<(), LaunchError> {
        if grid[1] == 1 && grid[2] == 1 && block[1] == 1 && block[2] == 1 {
            Ok(())
        } else {
            Err(LaunchError::new(""))
        }
    }
}
