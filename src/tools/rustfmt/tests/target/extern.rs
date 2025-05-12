// rustfmt-normalize_comments: true

extern crate foo;
extern crate foo as bar;

extern crate chrono;
extern crate dotenv;
extern crate futures;

extern crate bar;
extern crate foo;

// #2315
extern crate proc_macro;
extern crate proc_macro2;

// #3128
extern crate serde; // 1.0.78
extern crate serde_derive; // 1.0.78
extern crate serde_json; // 1.0.27

extern "C" {
    fn c_func(x: *mut *mut libc::c_void);

    fn c_func(
        x: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX,
        y: YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY,
    );

    #[test123]
    fn foo() -> uint64_t;

    pub fn bar();
}

extern "C" {
    fn DMR_GetDevice(
        pHDev: *mut HDEV,
        searchMode: DeviceSearchMode,
        pSearchString: *const c_char,
        devNr: c_uint,
        wildcard: c_char,
    ) -> TDMR_ERROR;

    fn quux() -> (); // Post comment

    pub type Foo;

    type Bar;
}

extern "Rust" {
    static ext: u32;
    // Some comment.
    pub static mut var: SomeType;
}

extern "C" {
    fn syscall(
        number: libc::c_long, // comment 1
        // comm 2
        ... // sup?
    ) -> libc::c_long;

    fn foo(x: *const c_char, ...) -> libc::c_long;
}

extern "C" {
    pub fn freopen(
        filename: *const c_char,
        mode: *const c_char,
        mode2: *const c_char,
        mode3: *const c_char,
        file: *mut FILE,
    ) -> *mut FILE;

    const fn foo() -> *mut Bar;
    unsafe fn foo() -> *mut Bar;

    pub(super) const fn foo() -> *mut Bar;
    pub(crate) unsafe fn foo() -> *mut Bar;
}

extern "C" {}

macro_rules! x {
    ($tt:tt) => {};
}

extern "macros" {
    x!(ident);
    x!(#);
    x![ident];
    x![#];
    x! {ident}
    x! {#}
}
