mod foo {
    #[link_section = ".rodata.STATIC"]
    #[used]
    static STATIC: [u32; 10] = [1; 10];
}

mod bar {
    #[no_mangle]
    extern "C" fn bar() -> i32 {
        0
    }
}
