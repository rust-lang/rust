mod foo {
    #[cfg_attr(target_os = "linux", link_section = ".rodata.STATIC")]
    #[cfg_attr(target_vendor = "apple", link_section = "__DATA,STATIC")]
    #[used]
    static STATIC: [u32; 10] = [1; 10];
}

mod bar {
    #[no_mangle]
    extern "C" fn bar() -> i32 {
        0
    }
}
