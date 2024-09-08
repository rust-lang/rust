//! Debugging aids.

/// Presence of a debugger. The debugger being concerned
/// is expected to use the OS API to debug this process.
#[derive(Copy, Clone, Debug)]
#[allow(unused)]
pub(crate) enum DebuggerPresence {
    /// The debugger is attached to this process.
    Detected,
    /// The debugger is not attached to this process.
    NotDetected,
}

#[cfg(target_os = "windows")]
mod os {
    use super::DebuggerPresence;

    #[link(name = "kernel32")]
    extern "system" {
        fn IsDebuggerPresent() -> i32;
    }

    pub(super) fn is_debugger_present() -> Option<DebuggerPresence> {
        // SAFETY: No state is shared between threads. The call reads
        // a field from the Thread Environment Block using the OS API
        // as required by the documentation.
        if unsafe { IsDebuggerPresent() } != 0 {
            Some(DebuggerPresence::Detected)
        } else {
            Some(DebuggerPresence::NotDetected)
        }
    }
}

#[cfg(any(target_vendor = "apple", target_os = "freebsd"))]
mod os {
    use libc::{c_int, sysctl, CTL_KERN, KERN_PROC, KERN_PROC_PID};

    use super::DebuggerPresence;
    use crate::io::{Cursor, Read, Seek, SeekFrom};
    use crate::process;

    const P_TRACED: i32 = 0x00000800;

    // The assumption is that the kernel structures available to the
    // user space may not shrink or repurpose the existing fields over
    // time. The kernels normally adhere to that for the backward
    // compatibility of the user space.

    // The macOS 14.5 SDK comes with a header `MacOSX14.5.sdk/usr/include/sys/sysctl.h`
    // that defines `struct kinfo_proc` be of `648` bytes on the 64-bit system. That has
    // not changed since macOS 10.13 (released in 2017) at least, validated by building
    // a C program in XCode while changing the build target. Apple provides this example
    // for reference: https://developer.apple.com/library/archive/qa/qa1361/_index.html.
    #[cfg(target_vendor = "apple")]
    const KINFO_PROC_SIZE: usize = if cfg!(target_pointer_width = "64") { 648 } else { 492 };
    #[cfg(target_vendor = "apple")]
    const KINFO_PROC_FLAGS_OFFSET: u64 = if cfg!(target_pointer_width = "64") { 32 } else { 16 };

    // Works for FreeBSD stable (13.3, 13.4) and current (14.0, 14.1).
    // The size of the structure has stayed the same for a long time,
    // at least since 2005:
    // https://lists.freebsd.org/pipermail/freebsd-stable/2005-November/019899.html
    #[cfg(target_os = "freebsd")]
    const KINFO_PROC_SIZE: usize = if cfg!(target_pointer_width = "64") { 1088 } else { 768 };
    #[cfg(target_os = "freebsd")]
    const KINFO_PROC_FLAGS_OFFSET: u64 = if cfg!(target_pointer_width = "64") { 368 } else { 296 };

    pub(super) fn is_debugger_present() -> Option<DebuggerPresence> {
        debug_assert_ne!(KINFO_PROC_SIZE, 0);

        let mut flags = [0u8; 4]; // `ki_flag` under FreeBSD and `p_flag` under macOS.
        let mut mib = [CTL_KERN, KERN_PROC, KERN_PROC_PID, process::id() as c_int];
        let mut info_size = KINFO_PROC_SIZE;
        let mut kinfo_proc = [0u8; KINFO_PROC_SIZE];

        // SAFETY: No state is shared with other threads. The sysctl call
        // is safe according to the documentation.
        if unsafe {
            sysctl(
                mib.as_mut_ptr(),
                mib.len() as u32,
                kinfo_proc.as_mut_ptr().cast(),
                &mut info_size,
                core::ptr::null_mut(),
                0,
            )
        } != 0
        {
            return None;
        }
        debug_assert_eq!(info_size, KINFO_PROC_SIZE);

        let mut reader = Cursor::new(kinfo_proc);
        reader.seek(SeekFrom::Start(KINFO_PROC_FLAGS_OFFSET)).ok()?;
        reader.read_exact(&mut flags).ok()?;
        // Just in case, not limiting this to the little-endian systems.
        let flags = i32::from_ne_bytes(flags);

        if flags & P_TRACED != 0 {
            Some(DebuggerPresence::Detected)
        } else {
            Some(DebuggerPresence::NotDetected)
        }
    }
}

#[cfg(target_os = "linux")]
mod os {
    use super::DebuggerPresence;
    use crate::fs::File;
    use crate::io::Read;

    pub(super) fn is_debugger_present() -> Option<DebuggerPresence> {
        // This function is crafted with the following goals:
        // * Memory efficiency: It avoids crashing the panicking process due to
        //   out-of-memory (OOM) conditions by not using large heap buffers or
        //   allocating significant stack space, which could lead to stack overflow.
        // * Minimal binary size: The function uses a minimal set of facilities
        //   from the standard library to avoid increasing the resulting binary size.
        //
        // To achieve these goals, the function does not use `[std::io::BufReader]`
        // and instead reads the file byte by byte using a sliding window approach.
        // It's important to note that the "/proc/self/status" pseudo-file is synthesized
        // by the Virtual File System (VFS), meaning it is not read from a slow or
        // non-volatile storage medium so buffering might not be as beneficial because
        // all data is read from memory, though this approach does incur a syscall for
        // each byte read.
        //
        // We cannot make assumptions about the file size or the position of the
        // target prefix ("TracerPid:"), so the function does not use
        // `[std::fs::read_to_string]` thus not employing UTF-8 to ASCII checking,
        // conversion, or parsing as we're looking for an ASCII prefix.
        //
        // These condiderations make the function deviate from the familiar concise pattern
        // of searching for a string in a text file.

        fn read_byte(file: &mut File) -> Option<u8> {
            let mut buffer = [0];
            file.read_exact(&mut buffer).ok()?;
            Some(buffer[0])
        }

        // The ASCII prefix of the datum we're interested in.
        const TRACER_PID: &[u8] = b"TracerPid:\t";

        let mut file = File::open("/proc/self/status").ok()?;
        let mut matched = 0;

        // Look for the `TRACER_PID` prefix.
        while let Some(byte) = read_byte(&mut file) {
            if byte == TRACER_PID[matched] {
                matched += 1;
                if matched == TRACER_PID.len() {
                    break;
                }
            } else {
                matched = 0;
            }
        }

        // Was the prefix found?
        if matched != TRACER_PID.len() {
            return None;
        }

        // It was; get the ASCII representation of the first digit
        // of the PID. That is enough to see if there is a debugger
        // attached as the kernel does not pad the PID on the left
        // with the leading zeroes.
        let byte = read_byte(&mut file)?;
        if byte.is_ascii_digit() && byte != b'0' {
            Some(DebuggerPresence::Detected)
        } else {
            Some(DebuggerPresence::NotDetected)
        }
    }
}

#[cfg(not(any(
    target_os = "windows",
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "linux"
)))]
mod os {
    pub(super) fn is_debugger_present() -> Option<super::DebuggerPresence> {
        None
    }
}

/// Detect the debugger presence.
///
/// The code does not try to detect the debugger at all costs (e.g., when anti-debugger
/// tricks are at play), it relies on the interfaces provided by the OS.
///
/// Return value:
/// * `None`: it's not possible to conclude whether the debugger is attached to this
///    process or not. When checking for the presence of the debugger, the detection logic
///    encountered an issue, such as the OS API throwing an error or the feature not being
///    implemented.
/// * `Some(DebuggerPresence::Detected)`: yes, the debugger is attached
///   to this process.
/// * `Some(DebuggerPresence::NotDetected)`: no, the debugger is not
///    attached to this process.
pub(crate) fn is_debugger_present() -> Option<DebuggerPresence> {
    if cfg!(miri) { None } else { os::is_debugger_present() }
}

/// Execute the breakpoint instruction if the debugger presence is detected.
/// Useful for breaking into the debugger without the need to set a breakpoint
/// in the debugger.
///
/// Note that there is a race between attaching or detaching the debugger, and running the
/// breakpoint instruction. This is nonetheless memory-safe, like [`crate::process::abort`]
/// is. In case the debugger is attached and the function is about
/// to run the breakpoint instruction yet right before that the debugger detaches, the
/// process will crash due to running the breakpoint instruction and the debugger not
/// handling the trap exception.
pub(crate) fn breakpoint_if_debugging() -> Option<DebuggerPresence> {
    let debugger_present = is_debugger_present();
    if let Some(DebuggerPresence::Detected) = debugger_present {
        // SAFETY: Executing the breakpoint instruction. No state is shared
        // or modified by this code.
        unsafe { core::intrinsics::breakpoint() };
    }

    debugger_present
}
