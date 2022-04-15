#![feature(deprecated_safe)]
#![feature(staged_api)]
#![stable(feature = "deprecated-safe-test", since = "1.61.0")]
#![warn(unused_unsafe)]

use std::ffi::OsStr;

#[stable(feature = "deprecated-safe-test", since = "1.61.0")]
#[deprecated_safe(since = "1.61.0", note = "reason")]
pub unsafe fn depr_safe() {}

#[stable(feature = "deprecated-safe-test", since = "1.61.0")]
#[deprecated_safe(since = "1.61.0", note = "reason")]
pub unsafe fn depr_safe_params(_: u32, _: u64) {}

#[stable(feature = "deprecated-safe-test", since = "1.61.0")]
#[deprecated_safe(since = "1.61.0", note = "reason")]
pub unsafe fn depr_safe_generic<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) {}

#[stable(feature = "deprecated-safe-test", since = "1.61.0")]
#[deprecated_safe(since = "1.61.0", note = "reason", unsafe_edition = "2015")]
pub unsafe fn depr_safe_2015() {}

#[stable(feature = "deprecated-safe-test", since = "1.61.0")]
#[deprecated_safe(since = "1.61.0", note = "reason", unsafe_edition = "2018")]
pub unsafe fn depr_safe_2018() {}

#[stable(feature = "deprecated-safe-test", since = "1.61.0")]
#[deprecated_safe(since = "1.61.0", note = "reason")]
pub unsafe trait DeprSafe {}

#[stable(feature = "deprecated-safe-test", since = "1.61.0")]
#[deprecated_safe(since = "1.61.0", note = "reason", unsafe_edition = "2015")]
pub unsafe trait DeprSafe2015 {}

#[stable(feature = "deprecated-safe-test", since = "1.61.0")]
#[deprecated_safe(since = "1.61.0", note = "reason", unsafe_edition = "2018")]
pub unsafe trait DeprSafe2018 {}

#[stable(feature = "deprecated-safe-test", since = "1.61.0")]
pub trait DeprSafeFns {
    #[stable(feature = "deprecated-safe-test", since = "1.61.0")]
    #[deprecated_safe(since = "1.61.0", note = "reason")]
    unsafe fn depr_safe_fn(&self) {}

    #[stable(feature = "deprecated-safe-test", since = "1.61.0")]
    #[deprecated_safe(since = "1.61.0", note = "reason")]
    unsafe fn depr_safe_params(&self, _: u32, _: u64) {}

    #[stable(feature = "deprecated-safe-test", since = "1.61.0")]
    #[deprecated_safe(since = "1.61.0", note = "reason")]
    unsafe fn depr_safe_fn_generic<K: AsRef<OsStr>, V: AsRef<OsStr>>(&self, key: K, value: V) {}

    #[stable(feature = "deprecated-safe-test", since = "1.61.0")]
    #[deprecated_safe(since = "1.61.0", note = "reason", unsafe_edition = "2015")]
    unsafe fn depr_safe_fn_2015(&self) {}

    #[stable(feature = "deprecated-safe-test", since = "1.61.0")]
    #[deprecated_safe(since = "1.61.0", note = "reason", unsafe_edition = "2018")]
    unsafe fn depr_safe_fn_2018(&self) {}
}
