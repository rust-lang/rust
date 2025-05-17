# `cfg_os_version_min`

The tracking issue for this feature is: [#136866]

[#136866]: https://github.com/rust-lang/rust/issues/136866

------------------------

The `cfg_os_version_min` feature makes it possible to conditionally compile
code depending on the target platform version(s).

As `cfg(os_version_min("platform", "version"))` is platform-specific, it
internally contains relevant `cfg(target_os = "platform")`. So specifying
`cfg(os_version_min("macos", "1.0"))` is equivalent to
`cfg(target_os = "macos")`.

Note that this only concerns the compile-time configured version; at runtime,
the version may be higher.


## Changing the version

Each Rust target has a default set of targetted platform versions. Examples
include the operating system version (`windows`, `macos`, Linux `kernel`) and
system library version (`libc`).

The mechanism for changing a targetted platform version is currently
target-specific (a more general mechanism may be created in the future):
- On macOS, you can select the minimum OS version using the
  `MACOSX_DEPLOYMENT_TARGET` environment variable. Similarly for iOS, tvOS,
  watchOS and visionOS, see the relevant [platform docs] for details.
- Others: Unknown.

[platform docs]: https://doc.rust-lang.org/nightly/rustc/platform-support.html


## Examples

Statically link the `preadv` symbol if known to be available on Apple
platforms and fall back to weak linking if not.

```rust,edition2021
#![feature(cfg_os_version_min)]
use core::ffi::c_int;
#
# #[allow(non_camel_case_types)]
# mod libc { // Fake libc for example
#     use core::ffi::{c_char, c_int, c_void};
#
#     pub type off_t = i64;
#     pub type iovec = c_void; // Fake typedef for example
#     pub type ssize_t = isize;
#
#     pub const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;
#
#     unsafe extern "C" {
#         pub unsafe fn dlsym(
#             handle: *mut c_void,
#             symbol: *const c_char,
#         ) -> *mut c_void;
#
#         pub unsafe fn preadv(
#             fd: c_int,
#             iovec: *const iovec,
#             n_iovec: c_int,
#             offset: off_t,
#         ) -> ssize_t;
#     }
# }
#
# // Only test Apple targets for now.
# #[cfg(target_vendor = "apple")] {

// Always available under these conditions.
#[cfg(any(
    os_version_min("macos", "11.0"),
    os_version_min("ios", "14.0"),
    os_version_min("tvos", "14.0"),
    os_version_min("watchos", "7.0"),
    os_version_min("visionos", "1.0"),
))]
let preadv = Some(libc::preadv);

// Otherwise, `preadv` needs to be weakly linked.
#[cfg(not(any(
    os_version_min("macos", "11.0"),
    os_version_min("ios", "14.0"),
    os_version_min("tvos", "14.0"),
    os_version_min("watchos", "7.0"),
    os_version_min("visionos", "1.0"),
)))]
let preadv = {
    // SAFETY: The C string is valid.
    let ptr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"preadv".as_ptr()) };
    type Fn = unsafe extern "C" fn(c_int, *const libc::iovec, c_int, libc::off_t) -> libc::ssize_t;
    // SAFETY: The function signature is correct, and the pointer is nullable.
    unsafe { core::mem::transmute::<*mut core::ffi::c_void, Option<Fn>>(ptr) }
};

if let Some(preadv) = preadv {
    # #[cfg(needs_to_pass_correct_arguments)]
    preadv(..) // Use preadv, it's available
} else {
    // ... fallback impl
}
# }
```
