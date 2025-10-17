use self::core_foundation::{
    CFDictionaryRef, CFHandle, CFIndex, CFStringRef, CFTypeRef, kCFAllocatorDefault,
    kCFPropertyListImmutable, kCFStringEncodingUTF8,
};
use crate::borrow::Cow;
use crate::bstr::ByteStr;
use crate::ffi::{CStr, c_char};
use crate::num::{NonZero, ParseIntError};
use crate::path::{Path, PathBuf};
use crate::ptr::null_mut;
use crate::sync::atomic::{AtomicU32, Ordering};
use crate::{env, fs};

mod core_foundation;
mod public_extern;
#[cfg(test)]
mod tests;

/// The version of the operating system.
///
/// We use a packed u32 here to allow for fast comparisons and to match Mach-O's `LC_BUILD_VERSION`.
type OSVersion = u32;

/// Combine parts of a version into an [`OSVersion`].
///
/// The size of the parts are inherently limited by Mach-O's `LC_BUILD_VERSION`.
#[inline]
const fn pack_os_version(major: u16, minor: u8, patch: u8) -> OSVersion {
    let (major, minor, patch) = (major as u32, minor as u32, patch as u32);
    (major << 16) | (minor << 8) | patch
}

/// [`pack_os_version`], but takes `i32` and saturates.
///
/// Instead of using e.g. `major as u16`, which truncates.
#[inline]
fn pack_i32_os_version(major: i32, minor: i32, patch: i32) -> OSVersion {
    let major: u16 = major.try_into().unwrap_or(u16::MAX);
    let minor: u8 = minor.try_into().unwrap_or(u8::MAX);
    let patch: u8 = patch.try_into().unwrap_or(u8::MAX);
    pack_os_version(major, minor, patch)
}

/// Get the current OS version, packed according to [`pack_os_version`].
///
/// # Semantics
///
/// The reported version on macOS might be 10.16 if the SDK version of the binary is less than 11.0.
/// This is a workaround that Apple implemented to handle applications that assumed that macOS
/// versions would always start with "10", see:
/// <https://github.com/apple-oss-distributions/xnu/blob/xnu-11215.81.4/libsyscall/wrappers/system-version-compat.c>
///
/// It _is_ possible to get the real version regardless of the SDK version of the binary, this is
/// what Zig does:
/// <https://github.com/ziglang/zig/blob/0.13.0/lib/std/zig/system/darwin/macos.zig>
///
/// We choose to not do that, and instead follow Apple's behaviour here, and return 10.16 when
/// compiled with an older SDK; the user should instead upgrade their tooling.
///
/// NOTE: `rustc` currently doesn't set the right SDK version when linking with ld64, so this will
/// have the wrong behaviour with `-Clinker=ld` on x86_64. But that's a `rustc` bug:
/// <https://github.com/rust-lang/rust/issues/129432>
#[inline]
fn current_version() -> OSVersion {
    // Cache the lookup for performance.
    //
    // 0.0.0 is never going to be a valid version ("vtool" reports "n/a" on 0 versions), so we use
    // that as our sentinel value.
    static CURRENT_VERSION: AtomicU32 = AtomicU32::new(0);

    // We use relaxed atomics instead of e.g. a `Once`, it doesn't matter if multiple threads end up
    // racing to read or write the version, `lookup_version` should be idempotent and always return
    // the same value.
    //
    // `compiler-rt` uses `dispatch_once`, but that's overkill for the reasons above.
    let version = CURRENT_VERSION.load(Ordering::Relaxed);
    if version == 0 {
        let version = lookup_version().get();
        CURRENT_VERSION.store(version, Ordering::Relaxed);
        version
    } else {
        version
    }
}

/// Look up the os version.
///
/// # Aborts
///
/// Aborts if reading or parsing the version fails (or if the system was out of memory).
///
/// We deliberately choose to abort, as having this silently return an invalid OS version would be
/// impossible for a user to debug.
// The lookup is costly and should be on the cold path because of the cache in `current_version`.
#[cold]
// Micro-optimization: We use `extern "C"` to abort on panic, allowing `current_version` (inlined)
// to be free of unwind handling. Aborting is required for `__isPlatformVersionAtLeast` anyhow.
extern "C" fn lookup_version() -> NonZero<OSVersion> {
    // Try to read from `sysctl` first (faster), but if that fails, fall back to reading the
    // property list (this is roughly what `_availability_version_check` does internally).
    let version = version_from_sysctl().unwrap_or_else(version_from_plist);

    // Use `NonZero` to try to make it clearer to the optimizer that this will never return 0.
    NonZero::new(version).expect("version cannot be 0.0.0")
}

/// Read the version from `kern.osproductversion` or `kern.iossupportversion`.
///
/// This is faster than `version_from_plist`, since it doesn't need to invoke `dlsym`.
fn version_from_sysctl() -> Option<OSVersion> {
    // This won't work in the simulator, as `kern.osproductversion` returns the host macOS version,
    // and `kern.iossupportversion` returns the host macOS' iOSSupportVersion (while you can run
    // simulators with many different iOS versions).
    if cfg!(target_abi = "sim") {
        // Fall back to `version_from_plist` on these targets.
        return None;
    }

    let sysctl_version = |name: &CStr| {
        let mut buf: [u8; 32] = [0; 32];
        let mut size = buf.len();
        let ptr = buf.as_mut_ptr().cast();
        let ret = unsafe { libc::sysctlbyname(name.as_ptr(), ptr, &mut size, null_mut(), 0) };
        if ret != 0 {
            // This sysctl is not available.
            return None;
        }
        let buf = &buf[..(size - 1)];

        if buf.is_empty() {
            // The buffer may be empty when using `kern.iossupportversion` on an actual iOS device,
            // or on visionOS when running under "Designed for iPad".
            //
            // In that case, fall back to `kern.osproductversion`.
            return None;
        }

        Some(parse_os_version(buf).unwrap_or_else(|err| {
            panic!("failed parsing version from sysctl ({}): {err}", ByteStr::new(buf))
        }))
    };

    // When `target_os = "ios"`, we may be in many different states:
    // - Native iOS device.
    // - iOS Simulator.
    // - Mac Catalyst.
    // - Mac + "Designed for iPad".
    // - Native visionOS device + "Designed for iPad".
    // - visionOS simulator + "Designed for iPad".
    //
    // Of these, only native, Mac Catalyst and simulators can be differentiated at compile-time
    // (with `target_abi = ""`, `target_abi = "macabi"` and `target_abi = "sim"` respectively).
    //
    // That is, "Designed for iPad" will act as iOS at compile-time, but the `ProductVersion` will
    // still be the host macOS or visionOS version.
    //
    // Furthermore, we can't even reliably differentiate between these at runtime, since
    // `dyld_get_active_platform` isn't publicly available.
    //
    // Fortunately, we won't need to know any of that; we can simply attempt to get the
    // `iOSSupportVersion` (which may be set on native iOS too, but then it will be set to the host
    // iOS version), and if that fails, fall back to the `ProductVersion`.
    if cfg!(target_os = "ios") {
        // https://github.com/apple-oss-distributions/xnu/blob/xnu-11215.81.4/bsd/kern/kern_sysctl.c#L2077-L2100
        if let Some(ios_support_version) = sysctl_version(c"kern.iossupportversion") {
            return Some(ios_support_version);
        }

        // On Mac Catalyst, if we failed looking up `iOSSupportVersion`, we don't want to
        // accidentally fall back to `ProductVersion`.
        if cfg!(target_abi = "macabi") {
            return None;
        }
    }

    // Introduced in macOS 10.13.4.
    // https://github.com/apple-oss-distributions/xnu/blob/xnu-11215.81.4/bsd/kern/kern_sysctl.c#L2015-L2051
    sysctl_version(c"kern.osproductversion")
}

/// Look up the current OS version(s) from `/System/Library/CoreServices/SystemVersion.plist`.
///
/// More specifically, from the `ProductVersion` and `iOSSupportVersion` keys, and from
/// `$IPHONE_SIMULATOR_ROOT/System/Library/CoreServices/SystemVersion.plist` on the simulator.
///
/// This file was introduced in macOS 10.3, which is well below the minimum supported version by
/// `rustc`, which is (at the time of writing) macOS 10.12.
///
/// # Implementation
///
/// We do roughly the same thing in here as `compiler-rt`, and dynamically look up CoreFoundation
/// utilities for parsing PLists (to avoid having to re-implement that in here, as pulling in a full
/// PList parser into `std` seems costly).
///
/// If this is found to be undesirable, we _could_ possibly hack it by parsing the PList manually
/// (it seems to use the plain-text "xml1" encoding/format in all versions), but that seems brittle.
fn version_from_plist() -> OSVersion {
    // Read `SystemVersion.plist`. Always present on Apple platforms, reading it cannot fail.
    let path = root_relative("/System/Library/CoreServices/SystemVersion.plist");
    let plist_buffer = fs::read(&path).unwrap_or_else(|e| panic!("failed reading {path:?}: {e}"));
    let cf_handle = CFHandle::new();
    parse_version_from_plist(&cf_handle, &plist_buffer)
}

/// Parse OS version from the given PList.
///
/// Split out from [`version_from_plist`] to allow for testing.
fn parse_version_from_plist(cf_handle: &CFHandle, plist_buffer: &[u8]) -> OSVersion {
    let plist_data = unsafe {
        cf_handle.CFDataCreateWithBytesNoCopy(
            kCFAllocatorDefault,
            plist_buffer.as_ptr(),
            plist_buffer.len() as CFIndex,
            cf_handle.kCFAllocatorNull(),
        )
    };
    assert!(!plist_data.is_null(), "failed creating CFData");
    let _plist_data_release = Deferred(|| unsafe { cf_handle.CFRelease(plist_data) });

    let plist = unsafe {
        cf_handle.CFPropertyListCreateWithData(
            kCFAllocatorDefault,
            plist_data,
            kCFPropertyListImmutable,
            null_mut(), // Don't care about the format of the PList.
            null_mut(), // Don't care about the error data.
        )
    };
    assert!(!plist.is_null(), "failed reading PList in SystemVersion.plist");
    let _plist_release = Deferred(|| unsafe { cf_handle.CFRelease(plist) });

    assert_eq!(
        unsafe { cf_handle.CFGetTypeID(plist) },
        unsafe { cf_handle.CFDictionaryGetTypeID() },
        "SystemVersion.plist did not contain a dictionary at the top level"
    );
    let plist: CFDictionaryRef = plist.cast();

    // Same logic as in `version_from_sysctl`.
    if cfg!(target_os = "ios") {
        if let Some(ios_support_version) =
            unsafe { string_version_key(cf_handle, plist, c"iOSSupportVersion") }
        {
            return ios_support_version;
        }

        // Force Mac Catalyst to use iOSSupportVersion (do not fall back to ProductVersion).
        if cfg!(target_abi = "macabi") {
            panic!("expected iOSSupportVersion in SystemVersion.plist");
        }
    }

    // On all other platforms, we can find the OS version by simply looking at `ProductVersion`.
    unsafe { string_version_key(cf_handle, plist, c"ProductVersion") }
        .expect("expected ProductVersion in SystemVersion.plist")
}

/// Look up a string key in a CFDictionary, and convert it to an [`OSVersion`].
unsafe fn string_version_key(
    cf_handle: &CFHandle,
    plist: CFDictionaryRef,
    lookup_key: &CStr,
) -> Option<OSVersion> {
    let cf_lookup_key = unsafe {
        cf_handle.CFStringCreateWithCStringNoCopy(
            kCFAllocatorDefault,
            lookup_key.as_ptr(),
            kCFStringEncodingUTF8,
            cf_handle.kCFAllocatorNull(),
        )
    };
    assert!(!cf_lookup_key.is_null(), "failed creating CFString");
    let _lookup_key_release = Deferred(|| unsafe { cf_handle.CFRelease(cf_lookup_key) });

    let value: CFTypeRef =
        unsafe { cf_handle.CFDictionaryGetValue(plist, cf_lookup_key) }.cast_mut();
    // `CFDictionaryGetValue` is a "getter", so we should not release,
    // the value is held alive internally by the CFDictionary, see:
    // https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmPractical.html#//apple_ref/doc/uid/TP40004447-SW12
    if value.is_null() {
        return None;
    }

    assert_eq!(
        unsafe { cf_handle.CFGetTypeID(value) },
        unsafe { cf_handle.CFStringGetTypeID() },
        "key in SystemVersion.plist must be a string"
    );
    let value: CFStringRef = value.cast();

    let mut version_str = [0u8; 32];
    let ret = unsafe {
        cf_handle.CFStringGetCString(
            value,
            version_str.as_mut_ptr().cast::<c_char>(),
            version_str.len() as CFIndex,
            kCFStringEncodingUTF8,
        )
    };
    assert_ne!(ret, 0, "failed getting string from CFString");

    let version_str =
        CStr::from_bytes_until_nul(&version_str).expect("failed converting CFString to CStr");

    Some(parse_os_version(version_str.to_bytes()).unwrap_or_else(|err| {
        panic!(
            "failed parsing version from PList ({}): {err}",
            ByteStr::new(version_str.to_bytes())
        )
    }))
}

/// Parse an OS version from a bytestring like b"10.1" or b"14.3.7".
fn parse_os_version(version: &[u8]) -> Result<OSVersion, ParseIntError> {
    if let Some((major, minor)) = version.split_once(|&b| b == b'.') {
        let major = u16::from_ascii(major)?;
        if let Some((minor, patch)) = minor.split_once(|&b| b == b'.') {
            let minor = u8::from_ascii(minor)?;
            let patch = u8::from_ascii(patch)?;
            Ok(pack_os_version(major, minor, patch))
        } else {
            let minor = u8::from_ascii(minor)?;
            Ok(pack_os_version(major, minor, 0))
        }
    } else {
        let major = u16::from_ascii(version)?;
        Ok(pack_os_version(major, 0, 0))
    }
}

/// Get a path relative to the root directory in which all files for the current env are located.
fn root_relative(path: &str) -> Cow<'_, Path> {
    if cfg!(target_abi = "sim") {
        let mut root = PathBuf::from(env::var_os("IPHONE_SIMULATOR_ROOT").expect(
            "environment variable `IPHONE_SIMULATOR_ROOT` must be set when executing under simulator",
        ));
        // Convert absolute path to relative path, to make the `.push` work as expected.
        root.push(Path::new(path).strip_prefix("/").unwrap());
        root.into()
    } else {
        Path::new(path).into()
    }
}

struct Deferred<F: FnMut()>(F);

impl<F: FnMut()> Drop for Deferred<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}
