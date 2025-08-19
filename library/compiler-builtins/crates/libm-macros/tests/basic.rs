#![feature(f16)]
#![feature(f128)]
// `STATUS_DLL_NOT_FOUND` on i686 MinGW, not worth looking into.
#![cfg(not(all(target_arch = "x86", target_os = "windows", target_env = "gnu")))]

macro_rules! basic {
    (
        fn_name: $fn_name:ident,
        FTy: $FTy:ty,
        CFn: $CFn:ty,
        CArgs: $CArgs:ty,
        CRet: $CRet:ty,
        RustFn: $RustFn:ty,
        RustArgs: $RustArgs:ty,
        RustRet: $RustRet:ty,
        public: $public:expr,
        attrs: [$($attr:meta),*],
        extra: [$($extra_tt:tt)*],
        fn_extra: $fn_extra:expr,
    ) => {
        $(#[$attr])*
        #[allow(dead_code)]
        pub mod $fn_name {
            type FTy= $FTy;
            type CFnTy<'a> = $CFn;
            type RustFnTy = $RustFn;
            type RustArgsTy = $RustArgs;
            type RustRetTy = $RustRet;
            const PUBLIC: bool = $public;
            const A: &[&str] = &[$($extra_tt)*];
            fn foo(a: f32) -> f32 {
                $fn_extra(a)
            }
        }
    };
}

mod test_basic {
    libm_macros::for_each_function! {
        callback: basic,
        emit_types: all,
        skip: [sin, cos],
        attributes: [
            // just some random attributes
            #[allow(clippy::pedantic)]
            #[allow(dead_code)]
            [sinf, cosf]
        ],
        extra: ["foo", "bar"],
        fn_extra: match MACRO_FN_NAME {
            sin => |x| x + 2.0,
            cos | cosf => |x: f32| x.MACRO_FN_NAME_NORMALIZED(),
            _ => |_x| 100.0
        }
    }
}

macro_rules! basic_no_extra {
    (
        fn_name: $fn_name:ident,
        attrs: [$($attr:meta),*],
    ) => {
        $(#[$attr])*
        mod $fn_name {}
    };
}

mod test_basic_no_extra {
    // Test with no extra, no skip, and no attributes
    libm_macros::for_each_function! {
        callback: basic_no_extra,
    }
}

mod test_only {
    // Test that only works
    libm_macros::for_each_function! {
        callback: basic_no_extra,
        only: [sin, sinf],
    }
}

macro_rules! specified_types {
    (
        fn_name: $fn_name:ident,
        RustFn: $RustFn:ty,
        RustArgs: $RustArgs:ty,
        attrs: [$($attr:meta),*],
    ) => {
        $(#[$attr])*
        #[allow(dead_code)]
        mod $fn_name {
            type RustFnTy = $RustFn;
            type RustArgsTy = $RustArgs;
        }
    };
}

mod test_emit_types {
    // Test that we can specify a couple types to emit
    libm_macros::for_each_function! {
        callback: specified_types,
        emit_types: [RustFn, RustArgs],
    }
}

#[test]
fn test_skip_f16_f128() {
    macro_rules! skip_f16_f128 {
        (
        fn_name: $fn_name:ident,
        attrs: [$($attr:meta),*],
        extra: $vec:ident,
    ) => {
            $vec.push(stringify!($fn_name));
        };
    }

    let mut v = Vec::new();
    // Test with no extra, no skip, and no attributes
    libm_macros::for_each_function! {
        callback: skip_f16_f128,
        skip_f16_f128: true,
        extra: v,
    }

    for name in v {
        assert!(!name.contains("f16"), "{name}");
        assert!(!name.contains("f128"), "{name}");
    }
}

#[test]
fn test_fn_extra_expansion() {
    macro_rules! fn_extra_expansion {
        (
            fn_name: $fn_name:ident,
            attrs: [$($attr:meta),*],
            fn_extra: $vec:expr,
        ) => {
            $vec.push(stringify!($fn_name));
        };
    }

    let mut vf16 = Vec::new();
    let mut vf32 = Vec::new();
    let mut vf64 = Vec::new();
    let mut vf128 = Vec::new();

    // Test with no extra, no skip, and no attributes
    libm_macros::for_each_function! {
        callback: fn_extra_expansion,
        fn_extra: match MACRO_FN_NAME {
            ALL_F16 => vf16,
            ALL_F32 => vf32,
            ALL_F64 => vf64,
            ALL_F128 => vf128,
        }
    }

    // Skip functions with a suffix after the type spec
    vf16.retain(|name| !name.ends_with("_r"));
    vf32.retain(|name| !name.ends_with("_r"));
    vf64.retain(|name| !name.ends_with("_r"));
    vf128.retain(|name| !name.ends_with("_r"));

    for name in vf16 {
        assert!(name.ends_with("f16"), "{name}");
    }
    for name in vf32 {
        assert!(name.ends_with("f"), "{name}");
    }
    let _ = vf64;
    for name in vf128 {
        assert!(name.ends_with("f128"), "{name}");
    }
}
