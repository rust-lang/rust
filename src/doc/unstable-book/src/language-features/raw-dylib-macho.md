# `raw_dylib_macho`

The tracking issue for this feature is: [#146356]

[#146356]: https://github.com/rust-lang/rust/issues/146356

------------------------

The `raw_dylib_macho` feature enables support for Mach-O/Darwin/Apple platforms
when using the `raw-dylib` linkage kind.

The `+verbatim` modifier currently must be set (though this restriction may be
lifted in the future).

```rust
//! Link to CoreFoundation without having to have the linker stubs available.
#![feature(raw_dylib_macho)]
#![cfg(target_vendor = "apple")]

#[link(
    name = "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation",
    kind = "raw-dylib",
    modifiers = "+verbatim",
)]
unsafe extern "C" {
    // Example function.
    safe fn CFRunLoopGetTypeID() -> core::ffi::c_ulong;
}

fn main() {
    let _ = CFRunLoopGetTypeID();
}
```

```rust
//! Weakly link to a symbol in libSystem.dylib that has been introduced
//! later than macOS 10.12 (which might mean that host tooling will have
//! trouble linking if it doesn't have the newer linker stubs available).
#![feature(raw_dylib_macho)]
#![feature(linkage)]
#![cfg(target_vendor = "apple")]

use std::ffi::{c_int, c_void};

#[link(name = "/usr/lib/libSystem.B.dylib", kind = "raw-dylib", modifiers = "+verbatim")]
unsafe extern "C" {
    #[linkage = "extern_weak"]
    safe static os_sync_wait_on_address: Option<unsafe extern "C" fn(*mut c_void, u64, usize, u32) -> c_int>;
}

fn main() {
    if let Some(_wait_on_address) = os_sync_wait_on_address {
        // Use new symbol
    } else {
        // Fallback implementation
    }
}
```

```rust,no_run
//! Link to a library relative to where the current binary will be installed,
//! without having that library available.
#![feature(raw_dylib_macho)]
#![cfg(target_vendor = "apple")]

#[link(name = "@executable_path/libfoo.dylib", kind = "raw-dylib", modifiers = "+verbatim")]
unsafe extern "C" {
    safe fn foo();
}

fn main() {
    foo();
}
```
