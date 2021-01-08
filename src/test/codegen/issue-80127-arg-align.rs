// ignore-tidy-linelength
// only-x86_64

// Regression test for issue 80127
// Tests that we properly propagate apply a `#[repr(align)]` attribute on a struct
// to external function definitions involving parameters with that struct type

#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Copy, Clone)]
pub struct bar {
    pub a: ::std::os::raw::c_ulong,
    pub b: ::std::os::raw::c_ulong,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct baz {
    pub a: bool,
    pub b: ::std::os::raw::c_uint,
}

extern "C" {
    // CHECK: declare i32 @foo_func(i8*, i8*, i8*, i64, i1 zeroext, i64, i8*, %bar* noalias nocapture byval(%bar) align 16 dereferenceable(16), i8*, i8*, i8*, i8*, i8*)
    pub fn foo_func(
        a: *mut ::std::os::raw::c_void,
        b: *mut ::std::os::raw::c_void,
        c: *const ::std::os::raw::c_char,
        d: ::std::os::raw::c_ulong,
        e: bool,
        f: baz,
        g: *mut ::std::os::raw::c_void,
        h: bar,
        i: *mut ::std::os::raw::c_void,
        j: *mut ::std::os::raw::c_void,
        k: *mut ::std::os::raw::c_void,
        l: *mut ::std::os::raw::c_void,
        m: *const ::std::os::raw::c_char,
    ) -> ::std::os::raw::c_int;
}

fn main() {
    let f = baz { a: true, b: 67 };
    let h = bar { a: 0, b: 99 };
    let m = std::ffi::CString::new("Hello, world").unwrap();
    unsafe {
        foo_func(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            12,
            true,
            f,
            std::ptr::null_mut(),
            h,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            m.as_ptr(),
        )
    };
}
