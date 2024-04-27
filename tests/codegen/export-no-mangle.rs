//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

mod private {
    // CHECK: @FOO =
    #[no_mangle]
    pub static FOO: u32 = 3;

    // CHECK: @BAR =
    #[export_name = "BAR"]
    static BAR: u32 = 3;

    // CHECK: void @a()
    #[no_mangle]
    pub extern "C" fn a() {}

    // CHECK: void @b()
    #[export_name = "b"]
    extern "C" fn b() {}

    // CHECK: void @c()
    #[export_name = "c"]
    #[inline]
    extern "C" fn c() {}

    // CHECK: void @d()
    #[export_name = "d"]
    #[inline(always)]
    extern "C" fn d() {}
}
