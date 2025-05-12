use super::alloc::*;
use super::time::*;
use crate::time::Duration;

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
fn epoch() {
    let t = r_efi::system::Time {
        year: 1970,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        nanosecond: 0,
        timezone: r_efi::efi::UNSPECIFIED_TIMEZONE,
        daylight: 0,
        pad1: 0,
        pad2: 0,
    };
    assert_eq!(system_time_internal::uefi_time_to_duration(t), Duration::new(0, 0));
}
