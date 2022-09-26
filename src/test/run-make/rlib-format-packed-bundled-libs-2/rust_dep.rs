#![feature(native_link_modifiers_verbatim)]
#[link(name = "native_dep.ext", kind = "static", modifiers = "+verbatim")]
extern "C" {
    fn native_f1() -> i32;
}

pub fn rust_dep() {
    unsafe {
        assert!(native_f1() == 1);
    }
}
