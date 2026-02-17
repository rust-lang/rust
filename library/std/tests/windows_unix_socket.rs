#![cfg(windows)]
#![cfg(not(miri))] // no socket support in Miri
#![feature(windows_unix_domain_sockets)]
// Now only test windows_unix_domain_sockets feature
// in the future, will test both unix and windows uds
use std::io::{Read, Write};
use std::os::windows::net::{UnixListener, UnixStream};
use std::{mem, thread};

macro_rules! skip_nonapplicable_oses {
    () => {
        // UDS have been available under Windows since Insider Preview Build
        // 17063. "Redstone 4" (RS4, version 1803, build number 17134) is
        // therefore the first official release to include it.
        if !is_windows_10_v1803_or_greater() {
            println!("Not running this test on too-old Windows.");
            return;
        }
    };
}

#[test]
fn win_uds_smoke_bind_connect() {
    skip_nonapplicable_oses!();

    let tmp = std::env::temp_dir();
    let sock_path = tmp.join("rust-test-uds-smoke.sock");
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path).expect("bind failed");
    let sock_path_clone = sock_path.clone();
    let tx = thread::spawn(move || {
        let mut stream = UnixStream::connect(&sock_path_clone).expect("connect failed");
        stream.write_all(b"hello").expect("write failed");
    });

    let (mut stream, _) = listener.accept().expect("accept failed");
    let mut buf = [0; 5];
    stream.read_exact(&mut buf).expect("read failed");
    assert_eq!(&buf, b"hello");

    tx.join().unwrap();

    drop(listener);
    let _ = std::fs::remove_file(&sock_path);
}

#[test]
fn win_uds_echo() {
    skip_nonapplicable_oses!();

    let tmp = std::env::temp_dir();
    let sock_path = tmp.join("rust-test-uds-echo.sock");
    let _ = std::fs::remove_file(&sock_path);

    let listener = UnixListener::bind(&sock_path).expect("bind failed");
    let srv = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept failed");
        let mut buf = [0u8; 128];
        loop {
            let n = match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => panic!("read error: {}", e),
            };
            stream.write_all(&buf[..n]).expect("write_all failed");
        }
    });

    let sock_path_clone = sock_path.clone();
    let cli = thread::spawn(move || {
        let mut stream = UnixStream::connect(&sock_path_clone).expect("connect failed");
        let req = b"hello windows uds";
        stream.write_all(req).expect("write failed");
        let mut resp = vec![0u8; req.len()];
        stream.read_exact(&mut resp).expect("read failed");
        assert_eq!(resp, req);
    });

    cli.join().unwrap();
    srv.join().unwrap();

    let _ = std::fs::remove_file(&sock_path);
}

#[test]
fn win_uds_path_too_long() {
    skip_nonapplicable_oses!();

    let tmp = std::env::temp_dir();
    let long_path = tmp.join("a".repeat(200));
    let result = UnixListener::bind(&long_path);
    assert!(result.is_err());
    let _ = std::fs::remove_file(&long_path);
}

#[test]
fn win_uds_existing_bind() {
    skip_nonapplicable_oses!();

    let tmp = std::env::temp_dir();
    let sock_path = tmp.join("rust-test-uds-existing.sock");
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path).expect("bind failed");
    let result = UnixListener::bind(&sock_path);
    assert!(result.is_err());
    drop(listener);
    let _ = std::fs::remove_file(&sock_path);
}

/// Returns true if we are currently running on Windows 10 v1803 (RS4) or greater.
fn is_windows_10_v1803_or_greater() -> bool {
    is_windows_version_greater_or_equal(NTDDI_WIN10_RS4)
}

/// Returns true if we are currently running on the given version of Windows
/// 10 (or newer).
fn is_windows_version_greater_or_equal(min_version: u32) -> bool {
    is_windows_version_or_greater(HIBYTE(OSVER(min_version)), LOBYTE(OSVER(min_version)), 0, 0)
}

/// Checks if we are running a version of Windows newer than the specified one.
fn is_windows_version_or_greater(
    major: u8,
    minor: u8,
    service_pack: u8,
    build_number: u32,
) -> bool {
    let mut osvi = OSVERSIONINFOEXW {
        dwOSVersionInfoSize: mem::size_of::<OSVERSIONINFOEXW>() as _,
        dwMajorVersion: u32::from(major),
        dwMinorVersion: u32::from(minor),
        wServicePackMajor: u16::from(service_pack),
        dwBuildNumber: build_number,
        ..OSVERSIONINFOEXW::default()
    };

    // SAFETY: this function is always safe to call.
    let condmask = unsafe {
        VerSetConditionMask(
            VerSetConditionMask(
                VerSetConditionMask(
                    VerSetConditionMask(0, VER_MAJORVERSION, VER_GREATER_EQUAL as _),
                    VER_MINORVERSION,
                    VER_GREATER_EQUAL as _,
                ),
                VER_SERVICEPACKMAJOR,
                VER_GREATER_EQUAL as _,
            ),
            VER_BUILDNUMBER,
            VER_GREATER_EQUAL as _,
        )
    };

    // SAFETY: osvi needs to point to a memory region valid for at least
    // dwOSVersionInfoSize bytes, which is the case here.
    (unsafe {
        RtlVerifyVersionInfo(
            &raw mut osvi,
            VER_MAJORVERSION | VER_MINORVERSION | VER_SERVICEPACKMAJOR,
            condmask,
        )
    }) == STATUS_SUCCESS
}

#[expect(non_snake_case)]
const fn HIBYTE(x: u16) -> u8 {
    ((x >> 8) & 0xFF) as u8
}

#[expect(non_snake_case)]
const fn LOBYTE(x: u16) -> u8 {
    (x & 0xFF) as u8
}

#[expect(non_snake_case)]
const fn OSVER(x: u32) -> u16 {
    ((x & OSVERSION_MASK) >> 16) as u16
}

// Inlined bindings because outside of `std` here.

type NTSTATUS = i32;
const STATUS_SUCCESS: NTSTATUS = 0;

#[expect(non_camel_case_types)]
type VER_FLAGS = u32;
const VER_BUILDNUMBER: VER_FLAGS = 4u32;
const VER_GREATER_EQUAL: VER_FLAGS = 3u32;
const VER_MAJORVERSION: VER_FLAGS = 2u32;
const VER_MINORVERSION: VER_FLAGS = 1u32;
const VER_SERVICEPACKMAJOR: VER_FLAGS = 32u32;

const OSVERSION_MASK: u32 = 4294901760u32;
const NTDDI_WIN10_RS4: u32 = 167772165u32;

#[expect(non_snake_case)]
#[repr(C)]
#[derive(Clone, Copy)]
struct OSVERSIONINFOEXW {
    pub dwOSVersionInfoSize: u32,
    pub dwMajorVersion: u32,
    pub dwMinorVersion: u32,
    pub dwBuildNumber: u32,
    pub dwPlatformId: u32,
    pub szCSDVersion: [u16; 128],
    pub wServicePackMajor: u16,
    pub wServicePackMinor: u16,
    pub wSuiteMask: u16,
    pub wProductType: u8,
    pub wReserved: u8,
}

impl Default for OSVERSIONINFOEXW {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

windows_link::link!("ntdll.dll" "system" fn RtlVerifyVersionInfo(versioninfo : *const OSVERSIONINFOEXW, typemask : u32, conditionmask : u64) -> NTSTATUS);
windows_link::link!("kernel32.dll" "system" fn VerSetConditionMask(conditionmask : u64, typemask : VER_FLAGS, condition : u8) -> u64);
