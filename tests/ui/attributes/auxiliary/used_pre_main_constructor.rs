//! Add a constructor that runs pre-main, similar to what the `ctor` crate does.
//!
//! #[ctor]
//! fn constructor() {
//!     println!("constructor");
//! }

//@ no-prefer-dynamic explicitly test with crates that are built as an archive
#![crate_type = "rlib"]

#[cfg_attr(
    any(
        target_os = "android",
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "haiku",
        target_os = "illumos",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "nto",
        target_os = "openbsd",
        target_os = "fuchsia",
        target_os = "managarm",
    ),
    link_section = ".init_array"
)]
#[cfg_attr(target_vendor = "apple", link_section = "__DATA,__mod_init_func,mod_init_funcs")]
#[cfg_attr(target_os = "windows", link_section = ".CRT$XCU")]
#[used]
static CONSTRUCTOR: extern "C" fn() = constructor;

#[cfg_attr(any(target_os = "linux", target_os = "android"), link_section = ".text.startup")]
extern "C" fn constructor() {
    println!("constructor");
}
