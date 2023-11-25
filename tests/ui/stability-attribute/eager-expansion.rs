// check-pass

#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "stable_test_feature", since = "3.3.3")]

macro_rules! sign_dependent_expr {
    (unsigned ? if unsigned { $unsigned_case:expr } if signed { $signed_case:expr }) => {
        $unsigned_case
    };
    (signed ? if unsigned { $unsigned_case:expr } if signed { $signed_case:expr }) => {
        $signed_case
    };
}

macro_rules! stable_feature {
    ($signedness:ident) => {
        sign_dependent_expr! {
            $signedness ?
            if unsigned { "nonzero" }
            if signed { "signed_nonzero" }
        }
    };
}

macro_rules! stable_since {
    ($signedness:ident) => {
        sign_dependent_expr! {
            $signedness ?
            if unsigned { "1.28.0" }
            if signed { "1.34.0" }
        }
    };
}

macro_rules! nonzero_integers {
    ($($signedness:ident $NonZero:ident($primitive:ty))*) => {
        $(
            #[stable(feature = stable_feature!($signedness), since = stable_since!($signedness))]
            pub struct $NonZero($primitive);
        )*
    };
}

nonzero_integers! {
    unsigned NonZeroU8(u8)
    unsigned NonZeroU16(u16)
    unsigned NonZeroU32(u32)
    unsigned NonZeroU64(u64)
    signed NonZeroI8(i8)
    signed NonZeroI16(i16)
    signed NonZeroI32(i32)
    signed NonZeroI64(i64)
}
