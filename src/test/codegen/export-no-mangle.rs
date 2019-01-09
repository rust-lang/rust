// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

mod private {
    // CHECK: @FOO =
    #[no_mangle]
    pub static FOO: u32 = 3;

    // CHECK: @BAR =
    #[export_name = "BAR"]
    static BAR: u32 = 3;

    // CHECK: void @foo()
    #[no_mangle]
    pub extern fn foo() {}

    // CHECK: void @bar()
    #[export_name = "bar"]
    extern fn bar() {}
}
