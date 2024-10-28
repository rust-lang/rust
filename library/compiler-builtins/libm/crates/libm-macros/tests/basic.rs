// `STATUS_DLL_NOT_FOUND` on i686 MinGW, not worth looking into.
#![cfg(not(all(target_arch = "x86", target_os = "windows", target_env = "gnu")))]

macro_rules! basic {
    (
        fn_name: $fn_name:ident,
        CFn: $CFn:ty,
        CArgs: $CArgs:ty,
        CRet: $CRet:ty,
        RustFn: $RustFn:ty,
        RustArgs: $RustArgs:ty,
        RustRet: $RustRet:ty,
        attrs: [$($meta:meta)*]
        extra: [$($extra_tt:tt)*],
        fn_extra: $fn_extra:expr,
    ) => {
        $(#[$meta])*
        mod $fn_name {
            #[allow(unused)]
            type CFnTy = $CFn;
            // type CArgsTy<'_> = $CArgs;
            // type CRetTy<'_> = $CRet;
            #[allow(unused)]
            type RustFnTy = $RustFn;
            #[allow(unused)]
            type RustArgsTy = $RustArgs;
            #[allow(unused)]
            type RustRetTy = $RustRet;
            #[allow(unused)]
            const A: &[&str] = &[$($extra_tt)*];
            #[allow(unused)]
            fn foo(a: f32) -> f32 {
                $fn_extra(a)
            }
        }
    };
}

mod test_basic {
    libm_macros::for_each_function! {
        callback: basic,
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
        CFn: $CFn:ty,
        CArgs: $CArgs:ty,
        CRet: $CRet:ty,
        RustFn: $RustFn:ty,
        RustArgs: $RustArgs:ty,
        RustRet: $RustRet:ty,
    ) => {
        mod $fn_name {
            #[allow(unused)]
            type CFnTy = $CFn;
            // type CArgsTy<'_> = $CArgs;
            // type CRetTy<'_> = $CRet;
            #[allow(unused)]
            type RustFnTy = $RustFn;
            #[allow(unused)]
            type RustArgsTy = $RustArgs;
            #[allow(unused)]
            type RustRetTy = $RustRet;
        }
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
