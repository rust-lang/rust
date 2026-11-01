use std::sync::atomic::{AtomicUsize, Ordering};

static COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn ctor<const N: usize>() {
    COUNT.fetch_add(N, Ordering::Relaxed);
}

#[rustfmt::skip]
macro_rules! ctor {
    ($ident:ident: $ty:ty = $ctor:expr) => {
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
        static $ident: $ty = $ctor;
    };
}

ctor! { CTOR1: unsafe extern "C" fn() = ctor::<1> }
ctor! { CTOR2: [unsafe extern "C" fn(); 2] = [ctor::<2>, ctor::<3>] }

fn main() {
    assert_eq!(COUNT.load(Ordering::Relaxed), 6, "ctors did not run");
}
