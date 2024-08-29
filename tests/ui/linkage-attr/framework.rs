// Check that linking frameworks on Apple platforms works.
//@ only-apple
//@ revisions: omit link weak both
//@ [omit]build-fail
//@ [link]run-pass
//@ [weak]run-pass
//@ [both]run-pass

// The linker's exact error output changes between Xcode versions, depends on
// linker invocation details, and the linker sometimes outputs more warnings.
//@ compare-output-lines-by-subset
//@ normalize-stderr-test: "linking with `.*` failed" -> "linking with `LINKER` failed"
//@ normalize-stderr-test: "Undefined symbols for architecture .*" -> "ld: Undefined symbols:"
//@ normalize-stderr-test: "._CFRunLoopGetTypeID.," -> "_CFRunLoopGetTypeID,"

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
