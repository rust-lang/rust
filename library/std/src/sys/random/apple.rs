//! Random data on non-macOS Apple platforms.
//!
//! Apple recommends the usage of `getentropy` in their security documentation[^1]
//! and mark it as being available in iOS 10.0, but we cannot use it on non-macOS
//! platforms as Apple in their *infinite wisdom* decided to consider this API
//! private, meaning its use will lead to App Store rejections (see #102643).
//!
//! Thus, we need to do the next best thing:
//!
//! Both `CCRandomGenerateBytes` and `SecRandomCopyBytes` simply call into
//! `CCRandomCopyBytes` with `kCCRandomDefault`. `CCRandomCopyBytes` manages a
//! CSPRNG which is seeded from the kernel's CSPRNG and which runs on its own
//! thread accessed via GCD (this is so wasteful...). Both are available on
//! iOS, but we use `CCRandomGenerateBytes` because it is accessible via
//! `libSystem` (libc) while the other needs to link to `Security.framework`.
//!
//! [^1]: <https://support.apple.com/en-gb/guide/security/seca0c73a75b/web>

pub fn fill_bytes(bytes: &mut [u8]) {
    let ret = unsafe { libc::CCRandomGenerateBytes(bytes.as_mut_ptr().cast(), bytes.len()) };
    assert_eq!(ret, libc::kCCSuccess, "failed to generate random data");
}
