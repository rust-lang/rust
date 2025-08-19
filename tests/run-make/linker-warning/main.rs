unsafe extern "C" {
    #[cfg(only_foo)]
    fn does_not_exist(p: *const u8) -> *const foo::Foo;
    #[cfg(not(only_foo))]
    fn does_not_exist(p: *const bar::Bar) -> *const foo::Foo;
}

fn main() {
    let _ = unsafe { does_not_exist(core::ptr::null()) };
}
