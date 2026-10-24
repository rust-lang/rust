// There's a lot of necessary redundancy in separator definition. Consolidated into a macro to
// prevent transcription errors.
macro_rules! path_separator_bytes {
    ($($sep:literal),+) => (
        pub const SEPARATORS: &[char] = &[$($sep as char,)+];
        pub const SEPARATORS_STR: &[&str] = &[$(
            match str::from_utf8(&[$sep]) {
                Ok(s) => s,
                Err(_) => panic!("path_separator_bytes must be ASCII bytes"),
            }
        ),+];

        #[inline]
        pub const fn is_sep_byte(b: u8) -> bool {
            $(b == $sep) ||+
        }
    )
}

cfg_select! {
    target_os = "windows" => {
        mod windows;
        mod windows_prefix;
        pub use windows::*;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        pub use sgx::*;
    }
    target_os = "solid_asp3" => {
        mod unsupported_backslash;
        pub use unsupported_backslash::*;
    }
    target_os = "uefi" => {
        mod uefi;
        pub use uefi::*;
    }
    target_os = "cygwin" => {
        mod cygwin;
        mod windows_prefix;
        pub use cygwin::*;
    }
    _ => {
        mod unix;
        pub use unix::*;
    }
}
