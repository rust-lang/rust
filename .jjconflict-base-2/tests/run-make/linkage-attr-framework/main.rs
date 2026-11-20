#![cfg_attr(any(weak, both), feature(link_arg_attribute))]

#[cfg_attr(any(link, both), link(name = "CoreFoundation", kind = "framework"))]
#[cfg_attr(
    any(weak, both),
    link(name = "-weak_framework", kind = "link-arg", modifiers = "+verbatim"),
    link(name = "CoreFoundation", kind = "link-arg", modifiers = "+verbatim")
)]
extern "C" {
    fn CFRunLoopGetTypeID() -> core::ffi::c_ulong;
}

fn main() {
    unsafe {
        CFRunLoopGetTypeID();
    }
}
