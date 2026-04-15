//! VFS Watch System ABI.
//!
//! Provides structures and constants for the kernel-native VFS watch system.
//! Watches allow userspace to receive asynchronous notifications about
//! filesystem changes (creations, deletions, modifications, etc.).

/// Event masks for watch registration and delivery.
pub mod mask {
    /// File was created in watched directory.
    pub const CREATE: u32 = 0x0000_0001;
    /// File was removed from watched directory.
    pub const REMOVE: u32 = 0x0000_0002;
    /// File content was modified.
    pub const MODIFY: u32 = 0x0000_0004;
    /// Metadata (mode, owner, etc.) was changed.
    pub const ATTRIB: u32 = 0x0000_0008;
    /// File was opened.
    pub const OPEN: u32 = 0x0000_0010;
    /// File was closed.
    pub const CLOSE: u32 = 0x0000_0020;
    /// File was moved from watched directory.
    pub const MOVE_FROM: u32 = 0x0000_0040;
    /// File was moved into watched directory.
    pub const MOVE_TO: u32 = 0x0000_0080;
    /// The watched object itself was deleted.
    pub const DELETE_SELF: u32 = 0x0000_0100;
    /// The watched object itself was moved.
    pub const MOVE_SELF: u32 = 0x0000_0200;

    /// Event queue overflowed (special event).
    pub const OVERFLOW: u32 = 0x0000_4000;
    /// File was unmounted.
    pub const UNMOUNT: u32 = 0x0000_2000;

    /// Helper for all move-related events.
    pub const MOVE: u32 = MOVE_FROM | MOVE_TO;
    /// Helper for all change-related events.
    pub const ALL_EVENTS: u32 = 0x0000_0FFF;
}

/// Flags for watch registration.
pub mod flags {
    /// Watch subdirectory changes recursively.
    pub const RECURSIVE: u32 = 0x0000_0001;
    /// Only watch if the path is a directory.
    pub const ONLYDIR: u32 = 0x0000_0002;
    /// remove watch after first event.
    pub const ONESHOT: u32 = 0x0000_0004;
    /// Do not follow symlinks.
    pub const NOFOLLOW: u32 = 0x0000_0008;
    /// Do not block on read if no events are available.
    pub const NONBLOCK: u32 = 0x0000_0010;
}

/// Event flags (returned in WatchEvent).
pub mod event_flags {
    /// Subject of the event is a directory.
    pub const IS_DIR: u16 = 0x0001;
}

/// A single watch event record.
///
/// Records are delivered via `read()` on a watch file descriptor.
/// Each record consists of this fixed-size header followed by `name_len` bytes
/// of the filename (if any).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct WatchEvent {
    /// The event mask (see [`mask`]).
    pub mask: u32,
    /// Event flags (see [`event_flags`]).
    pub flags: u16,
    /// Length of the filename immediately following this header.
    pub name_len: u16,
    /// Unique cookie linking paired events (e.g. MOVE_FROM/MOVE_TO).
    pub cookie: u32,
    /// Stable per-registration identifier for the node that was the subject of
    /// the event.
    ///
    /// This is a monotonically-increasing ID assigned by the kernel when a
    /// watch is registered (via `sys_watch_fd` / `sys_watch_path`).  It is
    /// unique for the lifetime of the watch and is never reused within a kernel
    /// session, so clients can reliably correlate events to the resource they
    /// registered without relying on raw inode numbers, which are not stable
    /// across remounts and can be recycled.
    pub subject_id: u64,
}

unsafe impl crate::wire::WireSafe for WatchEvent {}
