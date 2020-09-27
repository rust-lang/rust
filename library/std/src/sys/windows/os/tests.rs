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
