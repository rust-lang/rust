#![feature(abi_vectorcall)]
#![cfg_attr(target_arch = "x86", feature(raw_dylib))]

#[repr(C)]
#[derive(Clone)]
struct S {
    x: u8,
    y: i32,
}

#[repr(C)]
#[derive(Clone)]
struct S2 {
    x: i32,
    y: u8,
}

#[repr(C)]
#[derive(Clone)]
struct S3 {
    x: [u8; 5],
}

#[link(name = "extern", kind = "raw-dylib")]
extern "stdcall" {
    fn stdcall_fn_1(i: i32);
    fn stdcall_fn_2(c: u8, f: f32);
    fn stdcall_fn_3(d: f64);
    fn stdcall_fn_4(i: u8, j: u8, f: f32);
    fn stdcall_fn_5(a: S, b: i32);
    fn stdcall_fn_6(a: Option<&S>);
    fn stdcall_fn_7(a: S2, b: i32);
    fn stdcall_fn_8(a: S3, b: S3);
    fn stdcall_fn_9(x: u8, y: f64);
}

#[link(name = "extern", kind = "raw-dylib")]
extern "fastcall" {
    fn fastcall_fn_1(i: i32);
    fn fastcall_fn_2(c: u8, f: f32);
    fn fastcall_fn_3(d: f64);
    fn fastcall_fn_4(i: u8, j: u8, f: f32);
    fn fastcall_fn_5(a: S, b: i32);
    fn fastcall_fn_6(a: Option<&S>);
    fn fastcall_fn_7(a: S2, b: i32);
    fn fastcall_fn_8(a: S3, b: S3);
    fn fastcall_fn_9(x: u8, y: f64);
}

#[cfg(target_env = "msvc")]
#[link(name = "extern", kind = "raw-dylib")]
extern "vectorcall" {
    fn vectorcall_fn_1(i: i32);
    fn vectorcall_fn_2(c: u8, f: f32);
    fn vectorcall_fn_3(d: f64);
    fn vectorcall_fn_4(i: u8, j: u8, f: f32);
    fn vectorcall_fn_5(a: S, b: i32);
    fn vectorcall_fn_6(a: Option<&S>);
    fn vectorcall_fn_7(a: S2, b: i32);
    fn vectorcall_fn_8(a: S3, b: S3);
    fn vectorcall_fn_9(x: u8, y: f64);
}

pub fn library_function(run_msvc_only: bool) {
    unsafe {
        if !run_msvc_only {
            stdcall_fn_1(14);
            stdcall_fn_2(16, 3.5);
            stdcall_fn_3(3.5);
            stdcall_fn_4(1, 2, 3.0);
            stdcall_fn_5(S { x: 1, y: 2 }, 16);
            stdcall_fn_6(Some(&S { x: 10, y: 12 }));
            stdcall_fn_7(S2 { x: 15, y: 16 }, 3);
            stdcall_fn_8(S3 { x: [1, 2, 3, 4, 5] }, S3 { x: [6, 7, 8, 9, 10] });
            stdcall_fn_9(1, 3.0);

            fastcall_fn_1(14);
            fastcall_fn_2(16, 3.5);
            fastcall_fn_3(3.5);
            fastcall_fn_4(1, 2, 3.0);
            fastcall_fn_6(Some(&S { x: 10, y: 12 }));
            fastcall_fn_8(S3 { x: [1, 2, 3, 4, 5] }, S3 { x: [6, 7, 8, 9, 10] });
            fastcall_fn_9(1, 3.0);
        } else {
            // FIXME: 91167
            // rustc generates incorrect code for the calls to fastcall_fn_5 and fastcall_fn_7
            // on i686-pc-windows-gnu; disabling these until the indicated issue is fixed.
            fastcall_fn_5(S { x: 1, y: 2 }, 16);
            fastcall_fn_7(S2 { x: 15, y: 16 }, 3);

            // GCC doesn't support vectorcall: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=89485
            #[cfg(target_env = "msvc")]
            {
                vectorcall_fn_1(14);
                vectorcall_fn_2(16, 3.5);
                vectorcall_fn_3(3.5);
                vectorcall_fn_4(1, 2, 3.0);
                vectorcall_fn_5(S { x: 1, y: 2 }, 16);
                vectorcall_fn_6(Some(&S { x: 10, y: 12 }));
                vectorcall_fn_7(S2 { x: 15, y: 16 }, 3);
                vectorcall_fn_8(S3 { x: [1, 2, 3, 4, 5] }, S3 { x: [6, 7, 8, 9, 10] });
                vectorcall_fn_9(1, 3.0);
            }
        }
    }
}
