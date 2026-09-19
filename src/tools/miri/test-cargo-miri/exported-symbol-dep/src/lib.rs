#[no_mangle]
fn exported_symbol() -> i32 {
    123456
}

struct AssocFn;

impl AssocFn {
    #[no_mangle]
    fn assoc_fn_as_exported_symbol() -> i32 {
        -123456
    }
}

// Also check static constructors in dependencies are run.

#[rustfmt::skip]
#[macro_export]
macro_rules! ctor {
    ($ident:ident = $ctor:ident) => {
        #[cfg_attr(
            all(any(
                target_os = "linux",
                target_os = "android",
                target_os = "dragonfly",
                target_os = "freebsd",
                target_os = "haiku",
                target_os = "illumos",
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "solaris",
                target_os = "none",
                target_family = "wasm",
            )),
            link_section = ".init_array"
        )]
        #[cfg_attr(windows, link_section = ".CRT$XCU")]
        #[cfg_attr(
            any(target_os = "macos", target_os = "ios"),
            // We do not set the `mod_init_funcs` flag here since ctor/inventory also do not do
            // that. See <https://github.com/rust-lang/miri/pull/4459#discussion_r2200115629>.
            link_section = "__DATA,__mod_init_func"
        )]
        #[used]
        static $ident: unsafe extern "C" fn() = $ctor;
    };
}

static mut INITIALIZED: bool = false;

unsafe extern "C" fn ctor() {
    unsafe { INITIALIZED = true };
}

pub fn check_initialized() {
    assert!(unsafe { INITIALIZED });
}

ctor! { CTOR = ctor }
