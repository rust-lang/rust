//! This module defines the following.
//! - The `ErrCode` type.
//! - A constant for every error code, with a name like `E0123`.
//! - A static table `DIAGNOSTICS` pairing every error code constant with its
//!   long description text.

use std::fmt;
use std::sync::LazyLock;

use rustc_data_structures::fx::FxHashMap;

rustc_index::newtype_index! {
    #[max = 9999] // Because all error codes have four digits.
    #[orderable]
    #[encodable]
    #[debug_format = "ErrCode({})"]
    pub struct ErrCode {}
}

impl fmt::Display for ErrCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{:04}", self.as_u32())
    }
}

rustc_error_messages::into_diag_arg_using_display!(ErrCode);

macro_rules! define_error_code_constants_and_diagnostics_table {
    ($($num:literal,)*) => (
        $(
            pub const ${concat(E, $num)}: $crate::ErrCode = $crate::ErrCode::from_u32($num);
        )*
        static DIAGNOSTICS: LazyLock<FxHashMap<ErrCode, &'static str>> = LazyLock::new(|| {
            [
                $( (
                    ${concat(E, $num)},
                    include_str!(
                        concat!("../../rustc_error_codes/src/error_codes/E", stringify!($num), ".md")
                    )
                ), )*
            ]
            .iter()
            .copied()
            .collect()
        });
    )
}

rustc_error_codes::error_codes!(define_error_code_constants_and_diagnostics_table);

#[derive(Debug)]
pub struct InvalidErrorCode;

/// Returns `InvalidErrorCode` if the code requested does not exist.
pub fn try_find_description(code: ErrCode) -> Result<&'static str, InvalidErrorCode> {
    DIAGNOSTICS.get(&code).copied().ok_or(InvalidErrorCode)
}
