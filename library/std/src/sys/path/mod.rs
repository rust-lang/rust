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
        mod prefixed_paths;
        pub use windows::*;
        pub use prefixed_paths::*;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        mod nonprefixed_paths;
        pub use sgx::*;
        pub use nonprefixed_paths::*;
    }
    target_os = "solid_asp3" => {
        mod unsupported_backslash;
        mod prefixed_paths;
        pub use unsupported_backslash::*;
        pub use prefixed_paths::*;
    }
    target_os = "uefi" => {
        mod uefi;
        mod prefixed_paths;
        pub use uefi::*;
        pub use prefixed_paths::*;
    }
    target_os = "cygwin" => {
        mod cygwin;
        mod prefixed_paths;
        mod windows_prefix;
        pub use cygwin::*;
        pub use prefixed_paths::*;
    }
    _ => {
        mod unix;
        mod nonprefixed_paths;
        pub use unix::*;
        pub use nonprefixed_paths::*;
    }
}
