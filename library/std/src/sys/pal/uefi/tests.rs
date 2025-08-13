//! These tests are not run automatically right now. Please run these tests manually by copying them
//! to a separate project when modifying any related code.

use super::alloc::*;
use super::time::system_time_internal::{from_uefi, to_uefi};
use crate::io::{IoSlice, IoSliceMut};
use crate::time::Duration;

const SECS_IN_MINUTE: u64 = 60;

#[test]
fn align() {
    // UEFI ABI specifies that allocation alignment minimum is always 8. So this can be
    // statically verified.
    assert_eq!(POOL_ALIGNMENT, 8);

    // Loop over allocation-request sizes from 0-256 and alignments from 1-128, and verify
    // that in case of overalignment there is at least space for one additional pointer to
    // store in the allocation.
    for i in 0..256 {
        for j in &[1, 2, 4, 8, 16, 32, 64, 128] {
            if *j <= 8 {
                assert_eq!(align_size(i, *j), i);
            } else {
                assert!(align_size(i, *j) > i + size_of::<*mut ()>());
            }
        }
    }
}

#[test]
fn systemtime_start() {
    let t = r_efi::efi::Time {
        year: 1900,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        nanosecond: 0,
        timezone: -1440,
        daylight: 0,
        pad2: 0,
    };
    assert_eq!(from_uefi(&t), Duration::new(0, 0));
    assert_eq!(t, to_uefi(&from_uefi(&t), -1440, 0).unwrap());
    assert!(to_uefi(&from_uefi(&t), 0, 0).is_none());
}

#[test]
fn systemtime_utc_start() {
    let t = r_efi::efi::Time {
        year: 1900,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        pad1: 0,
        nanosecond: 0,
        timezone: 0,
        daylight: 0,
        pad2: 0,
    };
    assert_eq!(from_uefi(&t), Duration::new(1440 * SECS_IN_MINUTE, 0));
    assert_eq!(t, to_uefi(&from_uefi(&t), 0, 0).unwrap());
    assert!(to_uefi(&from_uefi(&t), -1440, 0).is_some());
}

#[test]
fn systemtime_end() {
    let t = r_efi::efi::Time {
        year: 9999,
        month: 12,
        day: 31,
        hour: 23,
        minute: 59,
        second: 59,
        pad1: 0,
        nanosecond: 0,
        timezone: 1440,
        daylight: 0,
        pad2: 0,
    };
    assert!(to_uefi(&from_uefi(&t), 1440, 0).is_some());
    assert!(to_uefi(&from_uefi(&t), 1439, 0).is_none());
}

// UEFI IoSlice and IoSliceMut Tests
//
// Strictly speaking, vectored read/write types for UDP4, UDP6, TCP4, TCP6 are defined
// separately in the UEFI Spec. However, they have the same signature. These tests just ensure
// that `IoSlice` and `IoSliceMut` are compatible with the vectored types for all the
// networking protocols.

unsafe fn to_slice<T>(val: &T) -> &[u8] {
    let len = size_of_val(val);
    unsafe { crate::slice::from_raw_parts(crate::ptr::from_ref(val).cast(), len) }
}

#[test]
fn io_slice_single() {
    let mut data = [0, 1, 2, 3, 4];

    let tcp4_frag = r_efi::protocols::tcp4::FragmentData {
        fragment_length: data.len().try_into().unwrap(),
        fragment_buffer: data.as_mut_ptr().cast(),
    };
    let tcp6_frag = r_efi::protocols::tcp6::FragmentData {
        fragment_length: data.len().try_into().unwrap(),
        fragment_buffer: data.as_mut_ptr().cast(),
    };
    let udp4_frag = r_efi::protocols::udp4::FragmentData {
        fragment_length: data.len().try_into().unwrap(),
        fragment_buffer: data.as_mut_ptr().cast(),
    };
    let udp6_frag = r_efi::protocols::udp6::FragmentData {
        fragment_length: data.len().try_into().unwrap(),
        fragment_buffer: data.as_mut_ptr().cast(),
    };
    let io_slice = IoSlice::new(&data);

    unsafe {
        assert_eq!(to_slice(&io_slice), to_slice(&tcp4_frag));
        assert_eq!(to_slice(&io_slice), to_slice(&tcp6_frag));
        assert_eq!(to_slice(&io_slice), to_slice(&udp4_frag));
        assert_eq!(to_slice(&io_slice), to_slice(&udp6_frag));
    }
}

#[test]
fn io_slice_mut_single() {
    let mut data = [0, 1, 2, 3, 4];

    let tcp4_frag = r_efi::protocols::tcp4::FragmentData {
        fragment_length: data.len().try_into().unwrap(),
        fragment_buffer: data.as_mut_ptr().cast(),
    };
    let tcp6_frag = r_efi::protocols::tcp6::FragmentData {
        fragment_length: data.len().try_into().unwrap(),
        fragment_buffer: data.as_mut_ptr().cast(),
    };
    let udp4_frag = r_efi::protocols::udp4::FragmentData {
        fragment_length: data.len().try_into().unwrap(),
        fragment_buffer: data.as_mut_ptr().cast(),
    };
    let udp6_frag = r_efi::protocols::udp6::FragmentData {
        fragment_length: data.len().try_into().unwrap(),
        fragment_buffer: data.as_mut_ptr().cast(),
    };
    let io_slice_mut = IoSliceMut::new(&mut data);

    unsafe {
        assert_eq!(to_slice(&io_slice_mut), to_slice(&tcp4_frag));
        assert_eq!(to_slice(&io_slice_mut), to_slice(&tcp6_frag));
        assert_eq!(to_slice(&io_slice_mut), to_slice(&udp4_frag));
        assert_eq!(to_slice(&io_slice_mut), to_slice(&udp6_frag));
    }
}

#[test]
fn io_slice_multi() {
    let mut data = [0, 1, 2, 3, 4];

    let tcp4_frag = r_efi::protocols::tcp4::FragmentData {
        fragment_length: data.len().try_into().unwrap(),
        fragment_buffer: data.as_mut_ptr().cast(),
    };
    let rhs =
        [tcp4_frag.clone(), tcp4_frag.clone(), tcp4_frag.clone(), tcp4_frag.clone(), tcp4_frag];
    let lhs = [
        IoSlice::new(&data),
        IoSlice::new(&data),
        IoSlice::new(&data),
        IoSlice::new(&data),
        IoSlice::new(&data),
    ];

    unsafe {
        assert_eq!(to_slice(&lhs), to_slice(&rhs));
    }
}

#[test]
fn io_slice_basic() {
    let data = [0, 1, 2, 3, 4];
    let mut io_slice = IoSlice::new(&data);

    assert_eq!(data, io_slice.as_slice());
    io_slice.advance(2);
    assert_eq!(&data[2..], io_slice.as_slice());
}

#[test]
fn io_slice_mut_basic() {
    let data = [0, 1, 2, 3, 4];
    let mut data_clone = [0, 1, 2, 3, 4];
    let mut io_slice_mut = IoSliceMut::new(&mut data_clone);

    assert_eq!(data, io_slice_mut.as_slice());
    assert_eq!(data, io_slice_mut.as_mut_slice());

    io_slice_mut.advance(2);
    assert_eq!(&data[2..], io_slice_mut.into_slice());
}
