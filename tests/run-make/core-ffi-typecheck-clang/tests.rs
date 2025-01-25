// tests.rs

use super::*; // `super` will include everything from `smallcore` once glued together

cfg_if! {
    if #[cfg(all(target_arch = "aarch64", target_abi = "ilp32"))] {
        // FIXME: long is not long enough on aarch64 ilp32, should be 8, defaulting to 4
        const XFAIL_C_LONG_SIZE: usize = 4;
        pub const TEST_C_LONG_SIZE: () = if size_of::<ffi::c_long>() != XFAIL_C_LONG_SIZE {
            panic!("wrong c_long size test ilp32");
        };
    }
    else {
        // Default test
        pub const TEST_C_LONG_SIZE: () = if size_of::<ffi::c_long>() != CLANG_C_LONG_SIZE {
            panic!("wrong c_long size");
        };
    }
}

cfg_if! {
    if #[cfg(target_arch = "csky")] {
        // FIXME: c_char signedness misallignment on csky, should be signed on CLANG
        const XFAIL_C_CHAR_SIGNED: bool = false;
        pub const TEST_C_CHAR_UNSIGNED: () = if ffi::c_char::SIGNED ^ XFAIL_C_CHAR_SIGNED {
            panic!("mismatched c_char signed, target_arch: csky");
        };
    }
    else if #[cfg(target_arch = "msp430")] {
        // FIXME: c_char signedness misallignment on msp430, should be signed on CLANG
        const XFAIL_C_CHAR_SIGNED: bool = false;  // Change to true for darwin
        pub const TEST_C_CHAR_UNSIGNED: () = if ffi::c_char::SIGNED ^ XFAIL_C_CHAR_SIGNED {
            panic!("mismatched c_char signed, target_arch: msp430");
        };
    }
    else {
        pub const TEST_C_CHAR_UNSIGNED: () = if ffi::c_char::SIGNED ^ CLANG_C_CHAR_SIGNED {
            panic!("mismatched c_char sign");
        };
    }
}

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

trait Signed {
    const SIGNED: bool;
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
