// compile-flags: -C no-prepopulate-passes

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
    pub extern fn a() {}

    // CHECK: void @b()
    #[export_name = "b"]
    extern fn b() {}

    // CHECK: void @c()
    #[export_name = "c"]
    #[inline]
    extern fn c() {}

    // CHECK: void @d()
    #[export_name = "d"]
    #[inline(always)]
    extern fn d() {}
}
