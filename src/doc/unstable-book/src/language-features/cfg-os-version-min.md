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

Statically link the `preadv` symbol if available, or fall back to weak linking if not.

```rust
#![feature(cfg_os_version_min)]
use libc;

// Always available under these conditions.
#[cfg(any(
    os_version_min("macos", "11.0"),
    os_version_min("ios", "14.0"),
    os_version_min("tvos", "14.0"),
    os_version_min("watchos", "7.0"),
    os_version_min("visionos", "1.0")
))]
let preadv = {
    extern "C" {
        fn preadv(libc::c_int, *const libc::iovec, libc::c_int, libc::off64_t) -> libc::ssize_t;
    }
    Some(preadv)
};

// Otherwise `preadv` needs to be weakly linked.
// We do that using a `weak!` macro, defined elsewhere.
#[cfg(not(any(
    os_version_min("macos", "11.0"),
    os_version_min("ios", "14.0"),
    os_version_min("tvos", "14.0"),
    os_version_min("watchos", "7.0"),
    os_version_min("visionos", "1.0")
)))]
weak!(fn preadv(libc::c_int, *const libc::iovec, libc::c_int, libc::off64_t) -> libc::ssize_t);

if let Some(preadv) = preadv {
    preadv(...) // Use preadv, it's available
} else {
    // ... fallback impl
}
```
