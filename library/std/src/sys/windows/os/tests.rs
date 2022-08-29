use crate::io::Error;
use crate::sys::c;

// tests `error_string` above
#[test]
fn ntstatus_error() {
    const STATUS_UNSUCCESSFUL: u32 = 0xc000_0001;
    assert!(
        !Error::from_raw_os_error((STATUS_UNSUCCESSFUL | c::FACILITY_NT_BIT) as _)
            .to_string()
            .contains("FormatMessageW() returned error")
    );
}

#[test]
fn smoketest_aligned_as() {
    use crate::{
        mem::{align_of, size_of},
        ptr::addr_of,
        sys::{c, AlignedAs},
    };
    type AlignedReparseBuf =
        AlignedAs<c::REPARSE_DATA_BUFFER, [u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE]>;
    assert!(size_of::<AlignedReparseBuf>() >= c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE);
    assert_eq!(align_of::<AlignedReparseBuf>(), align_of::<c::REPARSE_DATA_BUFFER>());
    let a = AlignedReparseBuf::new([0u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE]);
    // Quick and dirty offsetof check.
    assert_eq!(addr_of!(a).cast::<u8>(), addr_of!(a.value).cast::<u8>());
    // Smoke check that it's actually aligned.
    assert!(addr_of!(a.value).is_aligned_to(align_of::<c::REPARSE_DATA_BUFFER>()));
}
