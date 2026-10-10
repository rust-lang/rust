unsafe extern "C" fn ctor() -> i32 {
    //~^ERROR: calling a function with return type i32 passing return place of type ()
    0
}

#[rustfmt::skip]
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
        static $ident: unsafe extern "C" fn() -> i32 = $ctor;
    };
}

ctor! { CTOR = ctor }

fn main() {}
