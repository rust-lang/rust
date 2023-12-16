// Check that linking frameworks on Apple platforms works.
// only-macos
// revisions: omit link weak both
// [omit]build-fail
// [link]run-pass
// [weak]run-pass
// [both]run-pass
// normalize-stderr-test: "note: env .*" -> "note: [linker command]"
// normalize-stderr-test: "framework::main::\w* in framework\.framework\.\w*-cgu\.0\.rcgu\.o" -> "framework::main::HASH in framework.framework.HASH-cgu.0.rcgu.o"

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

pub fn main() {
    unsafe {
        CFRunLoopGetTypeID();
    }
}
