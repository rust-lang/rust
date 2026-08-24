use core::io::RawOsError;

/// On all targets to date, [`RawOsError`] is equivalent to a `c_int`, with the
/// notable exception of UEFI, where it is instead defined as `usize`.
#[test]
fn raw_os_error_ffi_guarantees() {
    let _: RawOsError = cfg_select! {
        target_os = "uefi" => 0 as usize,
        _ => 0 as core::ffi::c_int,
    };
}
