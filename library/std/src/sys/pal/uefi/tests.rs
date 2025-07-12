use super::alloc::*;
use super::time::system_time_internal::{from_uefi, to_uefi};
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
        pad1: 0,
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
