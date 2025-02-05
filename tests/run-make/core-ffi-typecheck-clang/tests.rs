/// This file compares the size and signedness of Rust `c_*` types to those from Clang,
/// based on the `CLANG_C_*` constants. Comparisons are done at compile time so this
/// does not need to be executed.
use super::*; // `super` will include everything from the common template

trait Signed {
    const SIGNED: bool;
}

// Verify Rust's 'c_long' is correct.
cfg_if! {
    if #[cfg(all(target_arch = "aarch64", target_abi = "ilp32"))] {
        // FIXME: long is not long enough on aarch64 ilp32, should be 8, defaulting to 4
        const XFAIL_C_LONG_SIZE: usize = 4;
        pub const TEST_C_LONG_SIZE: () = if size_of::<ffi::c_long>() != XFAIL_C_LONG_SIZE {
            panic!("mismatched c_long size, target_abi: ilp32");
        };
    }
    else if #[cfg(all(target_arch = "aarch64", target_os = "uefi"))] {
        // FIXME: c_long misallignment llvm target is aarch64-unknown-windows,
        // discrepencies between Rust target configuration and LLVM alias.
        const XFAIL_C_LONG_SIZE: usize = 8;
        pub const TEST_C_LONG_SIZE: () = if size_of::<ffi::c_long>() != XFAIL_C_LONG_SIZE {
            panic!("mismatched c_long size, target_os: uefi");
        };
    }
    else if #[cfg(all(target_arch = "x86_64", target_os = "uefi"))] {
        // FIXME: c_long misallignment llvm target is x86_64-unknown-windows,
        // discrepencies between Rust target configuration and LLVM alias.
        const XFAIL_C_LONG_SIZE: usize = 8;
        pub const TEST_C_LONG_SIZE: () = if size_of::<ffi::c_long>() != XFAIL_C_LONG_SIZE {
            panic!("mismatched c_long size, target_os: uefi");
        };
    }
    else {
        // Default test
        pub const TEST_C_LONG_SIZE: () = if size_of::<ffi::c_long>() != CLANG_C_LONG_SIZE {
            panic!("wrong c_long size");
        };
    }
}

// Verify Rust's 'c_char' has correct sign.
cfg_if! {
    if #[cfg(target_arch = "msp430")] {
        // FIXME: c_char signedness misallignment on msp430, should be signed on CLANG
        const XFAIL_C_CHAR_SIGNED: bool = false;
        pub const TEST_C_CHAR_UNSIGNED: () = if ffi::c_char::SIGNED != XFAIL_C_CHAR_SIGNED {
            panic!("mismatched c_char signed, target_arch: msp430");
        };
    }
    else if #[cfg(all(target_arch = "aarch64", target_os = "uefi"))] {
        // FIXME: c_char signedness misallignment llvm target is aarch64-unknown-windows,
        // discrepencies between Rust target configuration and LLVM alias.
        const XFAIL_C_CHAR_SIGNED: bool = false;
        pub const TEST_C_CHAR_UNSIGNED: () = if ffi::c_char::SIGNED != XFAIL_C_CHAR_SIGNED {
            panic!("mismatched c_char signed, target_os: uefi");
        };
    }
    else {
        pub const TEST_C_CHAR_UNSIGNED: () = if ffi::c_char::SIGNED != CLANG_C_CHAR_SIGNED {
            panic!("mismatched c_char sign");
        };
    }
}

// Verify Rust's 'c_double' is correct.
cfg_if! {
    if #[cfg(target_arch = "avr")] {
        // FIXME: double is not short enough on avr-unknown-gnu-atmega328 (should be 4 bytes)
        const XFAIL_C_DOUBLE_SIZE: usize = 8;
        pub const TEST_C_DOUBLE_SIZE: () = if size_of::<ffi::c_double>() != XFAIL_C_DOUBLE_SIZE {
            panic!("wrong c_double size, target_arch: avr");
        };
    }
    else {
        pub const TEST_C_DOUBLE_SIZE: () = if size_of::<ffi::c_double>() != CLANG_C_DOUBLE_SIZE {
            panic!("wrong c_double size");
        };
    }
}

// Verify Rust's 'c_size_t' is correct
cfg_if! {
    if #[cfg(all(target_arch = "aarch64", target_abi = "ilp32"))] {
        // FIXME: size_t is not short enough on aarch64 ilp32, should be 4, defaulting to 8
        const XFAIL_C_SIZE_T_SIZE: usize = 4;
        pub const TEST_C_SIZE_T_SIZE: () = if size_of::<ffi::c_size_t>() != XFAIL_C_SIZE_T_SIZE {
            panic!("wrong c_size_t size, target_arch: aarch64, target_abi: ilp32");
        };
    }
    else {
        pub const TEST_C_SIZE_T_SIZE: () = if size_of::<ffi::c_size_t>() != CLANG_C_SIZE_T_SIZE {
            panic!("wrong c_size_t size");
        };
    }
}

// Verify Rust's 'c_ssize_t' is correct
cfg_if! {
    if #[cfg(all(target_arch = "aarch64", target_abi = "ilp32"))] {
        // FIXME: ssize_t is not short enough on aarch64 ilp32, should be 4, defaulting to 8
        const XFAIL_C_SSIZE_T_SIZE: usize = 4;
        pub const TEST_C_SSIZE_T_SIZE: () = if size_of::<ffi::c_ssize_t>() != XFAIL_C_SSIZE_T_SIZE {
            panic!("wrong c_ssize_t size, target_arch: aarch64, target_abi: ilp32");
        };
    }
    else {
        pub const TEST_C_SSIZE_T_SIZE: () = if size_of::<ffi::c_ssize_t>() != CLANG_C_SIZE_T_SIZE {
            panic!("wrong c_size_t size");
        };
    }
}

// Verify Rust's 'c_ptrdiff_t' is correct
cfg_if! {
    if #[cfg(all(target_arch = "aarch64", target_abi = "ilp32"))] {
        // FIXME: c_ptrdiff_t is not short enough on aarch64 ilp32, should be 4, defaulting to 8
        const XFAIL_C_PTRDIFF_T_SIZE: usize = 4;
        pub const TEST_C_PTRDIFF_T_SIZE: () =
            if size_of::<ffi::c_ptrdiff_t>() != XFAIL_C_PTRDIFF_T_SIZE {
            panic!("wrong c_ssize_t size, target_arch: aarch64, target_abi: ilp32");
        };
    }
    else {
        pub const TEST_C_PTRDIFF_T_SIZE: () =
            if size_of::<ffi::c_ptrdiff_t>() != CLANG_C_PTRDIFF_T_SIZE {
            panic!("wrong c_size_t size");
        };
    }
}

impl Signed for i8 {
    const SIGNED: bool = true;
}

impl Signed for u8 {
    const SIGNED: bool = false;
}

//c_char size
pub const TEST_C_CHAR_SIZE: () = if size_of::<ffi::c_char>() != CLANG_C_CHAR_SIZE {
    panic!("wrong c_char size");
};

//c_int size
pub const TEST_C_INT_SIZE: () = if size_of::<ffi::c_int>() != CLANG_C_INT_SIZE {
    panic!("mismatched c_int size");
};

//c_short size
pub const TEST_C_SHORT_SIZE: () = if size_of::<ffi::c_short>() != CLANG_C_SHORT_SIZE {
    panic!("wrong c_short size");
};

//c_longlong size
pub const TEST_C_LONGLONG_SIZE: () = if size_of::<ffi::c_longlong>() != CLANG_C_LONGLONG_SIZE {
    panic!("wrong c_longlong size");
};

//c_float size
pub const TEST_C_FLOAT_SIZE: () = if size_of::<ffi::c_float>() != CLANG_C_FLOAT_SIZE {
    panic!("wrong c_float size");
};

//c_schar size
pub const TEST_C_SCHAR_SIZE: () = if size_of::<ffi::c_schar>() != CLANG_C_CHAR_SIZE {
    panic!("wrong c_schar size");
};

//c_uchar size
pub const TEST_C_UCHAR_SIZE: () = if size_of::<ffi::c_uchar>() != CLANG_C_CHAR_SIZE {
    panic!("wrong c_uchar size");
};

//c_uint size
pub const TEST_C_UINT_SIZE: () = if size_of::<ffi::c_int>() != CLANG_C_INT_SIZE {
    panic!("mismatched c_uint size");
};

//c_ushort size
pub const TEST_C_USHORT_SIZE: () = if size_of::<ffi::c_short>() != CLANG_C_SHORT_SIZE {
    panic!("wrong c_ushort size");
};

//c_ulonglong size
pub const TEST_C_ULONGLONG_SIZE: () = if size_of::<ffi::c_ulonglong>() != CLANG_C_LONGLONG_SIZE {
    panic!("wrong c_ulonglong size");
};
