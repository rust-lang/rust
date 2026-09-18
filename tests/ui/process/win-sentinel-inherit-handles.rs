// Tests `inherit_handles` by object identity instead of handle count:
// the parent records a sentinel file's identity and the child checks
// whether the numeric handle it received addresses the same object.
// A count delta cannot prove WHICH handle crossed the boundary, and a
// reused numeric value can look "valid" while addressing another object.

//@ run-pass
//@ only-windows
//@ needs-subprocess
//@ edition: 2024

#![feature(windows_process_extensions_inherit_handles)]

use std::ffi::OsString;
use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

fn main() {
    // Parent runs bare; child runs as `--child <mode> <hex> <vol> <hi> <lo>`.
    let mut args = std::env::args_os();
    let _exe = args.next();
    let rest: Vec<OsString> = args.collect();
    if rest.first().is_some_and(|s| s == "--child") {
        let mode = rest.get(1).expect("child case mode").clone();
        child(&mode, rest[2..].to_vec());
    } else {
        assert!(rest.is_empty(), "unexpected arguments");
        parent();
    }
}

fn parent() {
    run_case("default-inherit", true, false);
    run_case("no-inherit", true, true);
    run_case("not-inheritable", false, false);
}

static CASE_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Live sentinel: the open file (kept alive across the spawn), its
/// numeric value for argv, and its file identity for the comparison.
struct Sentinel {
    _file: std::fs::File,
    hex: String,
    id: (u32, u32, u32),
    path: std::path::PathBuf,
}

impl Sentinel {
    fn create(tag: &str, inheritable: bool) -> Self {
        let path = std::env::temp_dir().join(format!(
            "rust-sentinel-inherit-{}-{}-{}.tmp",
            std::process::id(),
            tag,
            CASE_COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        let file = std::fs::File::create(&path).expect("create sentinel file");
        unsafe {
            let ok = winapi::SetHandleInformation(
                file.as_raw_handle() as _,
                winapi::HANDLE_FLAG_INHERIT,
                if inheritable { winapi::HANDLE_FLAG_INHERIT } else { 0 },
            );
            assert_ne!(ok, 0, "SetHandleInformation failed");
            let id = winapi::file_id(file.as_raw_handle() as _).expect("file identity");
            let hex = format!("{:p}", file.as_raw_handle());
            Sentinel { _file: file, hex, id, path }
        }
    }
}

impl Drop for Sentinel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn run_case(tag: &str, inheritable: bool, no_inherit: bool) {
    let sentinel = Sentinel::create(tag, inheritable);
    let (vol, hi, lo) = sentinel.id;
    let mut cmd = Command::new(&std::env::current_exe().unwrap());
    cmd.arg("--child")
        .arg(tag)
        .arg(&sentinel.hex)
        .arg(vol.to_string())
        .arg(hi.to_string())
        .arg(lo.to_string());
    if no_inherit {
        cmd.inherit_handles(false);
    }
    let status = cmd.spawn().expect("spawn child").wait().expect("wait child");
    assert!(status.success(), "case {tag} failed: child exit = {:?}", status.code());
    drop(sentinel);
}

fn child(mode: &std::ffi::OsStr, rest: Vec<OsString>) {
    let mode = mode.to_string_lossy().into_owned();
    assert_eq!(rest.len(), 4, "child usage: --child <mode> <hex> <vol> <hi> <lo>");
    let str_at = |i: usize| rest[i].to_string_lossy().into_owned();
    let value = usize::from_str_radix(str_at(0).trim_start_matches("0x"), 16).expect("hex handle");
    let expected = (
        str_at(1).parse::<u32>().expect("vol"),
        str_at(2).parse::<u32>().expect("hi"),
        str_at(3).parse::<u32>().expect("lo"),
    );
    let observed = unsafe { winapi::file_id(value as _) };
    // Compare identities, never mere validity: a numerically reused
    // value addressing a different object must not pass.
    let same_object = observed == Some(expected);
    let pass = match mode.as_str() {
        "default-inherit" => same_object,
        "no-inherit" | "not-inheritable" => !same_object,
        other => panic!("unknown child mode: {other}"),
    };
    if !pass {
        eprintln!("mode={mode} same_object={same_object} observed={observed:?}");
        std::process::exit(1);
    }
}

// Minimal kernel32 surface for the test (same pattern as the existing
// win-inherit-handles.rs `winapi` module).
mod winapi {
    pub const HANDLE_FLAG_INHERIT: u32 = 1;

    #[repr(C)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    #[repr(C)]
    struct ByHandleFileInformation {
        attributes: u32,
        creation: FileTime,
        last_access: FileTime,
        last_write: FileTime,
        volume: u32,
        _size_high: u32,
        _size_low: u32,
        _links: u32,
        index_high: u32,
        index_low: u32,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn SetHandleInformation(h: *mut std::ffi::c_void, mask: u32, flags: u32) -> i32;
        fn GetFileInformationByHandle(
            h: *mut std::ffi::c_void,
            info: *mut ByHandleFileInformation,
        ) -> i32;
    }

    /// (volume, index_high, index_low) of the file behind `h`, or None.
    pub unsafe fn file_id(h: *mut std::ffi::c_void) -> Option<(u32, u32, u32)> {
        // SAFETY: all-zeros is valid for this plain-data struct, and `h`
        // is either our own open file or a numeric probe the caller owns.
        let mut info: ByHandleFileInformation = unsafe { std::mem::zeroed() };
        if unsafe { GetFileInformationByHandle(h, &mut info) } == 0 {
            return None;
        }
        Some((info.volume, info.index_high, info.index_low))
    }
}
