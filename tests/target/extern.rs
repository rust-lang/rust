
extern {
    fn c_func(x: *mut *mut libc::c_void);

    fn c_func(x: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX,
              y: YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY);

    #[test123]
    fn foo() -> uint64_t;

    pub fn bar();
}

extern {
    fn DMR_GetDevice(pHDev: *mut HDEV,
                     searchMode: DeviceSearchMode,
                     pSearchString: *const c_char,
                     devNr: c_uint,
                     wildcard: c_char)
                     -> TDMR_ERROR;

    fn quux() -> (); // Post comment
}

extern "Rust" {
    static ext: u32;
    // Some comment.
    pub static mut var: SomeType;
}
