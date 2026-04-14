//! Per-process VFS file descriptor table.
//!
//! Each process that uses VFS syscalls has an `FdTable` embedded inside its
//! `ProcessInfo`.  Kernel threads do not have an `FdTable` and all VFS
//! syscalls return `ENOENT` for them.
//!
//! File descriptors are non-negative integers starting at 0.  Descriptors 0,
//! 1, and 2 are **stdio** (stdin / stdout / stderr) and are pre-populated at
//! process spawn time via [`FdTable::insert_at`].  All other descriptors are
//! allocated by [`FdTable::open`] which scans for the lowest free slot.
//!
//! # Limits
//! `MAX_FDS` open files per process.  This is intentionally small for now.

use abi::errors::{Errno, SysResult};
use alloc::string::String;
use alloc::sync::Arc;
use spin::Mutex;

use super::{OpenFlags, VfsNode};

pub const MAX_FDS: usize = 256;
pub const FD_CLOEXEC: u32 = abi::syscall::fd_flags::FD_CLOEXEC;

/// A single open-file entry in the FD table.
///
/// The `offset` is wrapped in `Arc<Mutex<u64>>` so that file descriptors
/// created by `dup` or `dup2` share the same file position, matching POSIX
/// open-file-description semantics.
#[derive(Clone)]
pub struct OpenFile {
    pub node: Arc<dyn VfsNode>,
    pub status_flags: Arc<Mutex<OpenFlags>>,
    pub fd_flags: u32,
    /// Absolute path of this open file (if known).
    pub path: Arc<String>,
    /// Shared read/write position — cloned (not copied) on dup/dup2.
    pub offset: Arc<Mutex<u64>>,
}

/// Per-process file descriptor table.
#[derive(Clone)]
pub struct FdTable {
    entries: alloc::vec::Vec<Option<OpenFile>>,
}

impl FdTable {
    pub fn new() -> Self {
        let mut entries = alloc::vec::Vec::with_capacity(MAX_FDS);
        for _ in 0..MAX_FDS {
            entries.push(None);
        }
        Self { entries }
    }

    /// Insert `node` into the table and return the allocated file descriptor.
    ///
    /// Scans from slot 0 and returns the first free slot.  Because slots 0–2
    /// are pre-populated at spawn time, regular opens naturally receive fd ≥ 3.
    /// Returns `EMFILE` when all slots are exhausted.
    pub fn open(
        &mut self,
        node: Arc<dyn VfsNode>,
        flags: OpenFlags,
        path: String,
    ) -> SysResult<u32> {
        for i in 0..MAX_FDS {
            if self.entries[i].is_none() {
                self.entries[i] = Some(OpenFile {
                    node,
                    status_flags: Arc::new(Mutex::new(flags)),
                    fd_flags: 0,
                    path: Arc::new(path),
                    offset: Arc::new(Mutex::new(0)),
                });
                return Ok(i as u32);
            }
        }
        Err(Errno::EMFILE)
    }

    /// Insert `node` at a specific file descriptor slot.
    ///
    /// Used at process creation to populate stdin (0), stdout (1), stderr (2).
    /// Returns `EBADF` if `fd` is out of range or already occupied.
    pub fn insert_at(
        &mut self,
        fd: u32,
        node: Arc<dyn VfsNode>,
        flags: OpenFlags,
        path: String,
    ) -> SysResult<()> {
        let idx = fd as usize;
        if idx >= MAX_FDS {
            return Err(Errno::EBADF);
        }
        if self.entries[idx].is_some() {
            return Err(Errno::EBADF);
        }
        self.entries[idx] = Some(OpenFile {
            node,
            status_flags: Arc::new(Mutex::new(flags)),
            fd_flags: 0,
            path: Arc::new(path),
            offset: Arc::new(Mutex::new(0)),
        });
        Ok(())
    }

    /// Duplicate `old_fd` to the lowest available descriptor.
    ///
    /// The new descriptor shares the same `VfsNode` **and** file offset as
    /// `old_fd`, matching POSIX open-file-description semantics.
    /// Returns the new file descriptor, or `EBADF` if `old_fd` is not open.
    pub fn dup(&mut self, old_fd: u32) -> SysResult<u32> {
        let idx = old_fd as usize;
        if idx >= MAX_FDS {
            return Err(Errno::EBADF);
        }
        let entry = self.entries[idx].as_ref().ok_or(Errno::EBADF)?;
        let new_node = entry.node.clone();
        let new_status_flags = entry.status_flags.clone();
        let new_path = entry.path.clone();
        let shared_offset = entry.offset.clone(); // share offset with original
        for i in 0..MAX_FDS {
            if self.entries[i].is_none() {
                self.entries[i] = Some(OpenFile {
                    node: new_node,
                    status_flags: new_status_flags,
                    fd_flags: 0,
                    path: new_path,
                    offset: shared_offset,
                });
                return Ok(i as u32);
            }
        }
        Err(Errno::EMFILE)
    }

    /// Duplicate `old_fd` to `new_fd`.
    ///
    /// If `new_fd` is already open it is closed first.  If `old_fd == new_fd`
    /// this is a no-op that returns `new_fd`.  Both descriptors will share the
    /// same file offset after this call.  Returns `EBADF` if `old_fd` is not
    /// open or either fd is out of range.
    pub fn dup2(&mut self, old_fd: u32, new_fd: u32) -> SysResult<u32> {
        let old_idx = old_fd as usize;
        let new_idx = new_fd as usize;
        if old_idx >= MAX_FDS || new_idx >= MAX_FDS {
            return Err(Errno::EBADF);
        }
        if old_fd == new_fd {
            // Verify old_fd is open.
            self.entries[old_idx].as_ref().ok_or(Errno::EBADF)?;
            return Ok(new_fd);
        }
        let entry = self.entries[old_idx].as_ref().ok_or(Errno::EBADF)?;
        let new_node = entry.node.clone();
        let new_status_flags = entry.status_flags.clone();
        let new_path = entry.path.clone();
        let shared_offset = entry.offset.clone(); // share offset with original
        // Close new_fd if open.
        if let Some(old_entry) = self.entries[new_idx].take() {
            old_entry.node.close();
        }
        self.entries[new_idx] = Some(OpenFile {
            node: new_node,
            status_flags: new_status_flags,
            fd_flags: 0,
            path: new_path,
            offset: shared_offset,
        });
        Ok(new_fd)
    }

    /// Return a reference to the open-file entry for `fd`, or `EBADF`.
    pub fn get(&self, fd: u32) -> SysResult<&OpenFile> {
        let idx = fd as usize;
        if idx >= MAX_FDS {
            return Err(Errno::EBADF);
        }
        self.entries[idx].as_ref().ok_or(Errno::EBADF)
    }

    /// Return a mutable reference to the open-file entry for `fd`, or `EBADF`.
    pub fn get_mut(&mut self, fd: u32) -> SysResult<&mut OpenFile> {
        let idx = fd as usize;
        if idx >= MAX_FDS {
            return Err(Errno::EBADF);
        }
        self.entries[idx].as_mut().ok_or(Errno::EBADF)
    }

    /// Read the descriptor flags (`FD_*`) for `fd`.
    pub fn get_fd_flags(&self, fd: u32) -> SysResult<u32> {
        Ok(self.get(fd)?.fd_flags)
    }

    /// Replace the descriptor flags (`FD_*`) for `fd`.
    pub fn set_fd_flags(&mut self, fd: u32, flags: u32) -> SysResult<()> {
        self.get_mut(fd)?.fd_flags = flags & FD_CLOEXEC;
        Ok(())
    }

    /// Close file descriptor `fd`.  Returns `EBADF` if not open.
    pub fn close(&mut self, fd: u32) -> SysResult<()> {
        let idx = fd as usize;
        if idx >= MAX_FDS {
            return Err(Errno::EBADF);
        }
        let entry = self.entries[idx].take().ok_or(Errno::EBADF)?;
        entry.node.close();
        Ok(())
    }

    /// Close all open file descriptors (called on process exit).
    pub fn close_all(&mut self) {
        for slot in self.entries.iter_mut() {
            if let Some(entry) = slot.take() {
                entry.node.close();
            }
        }
    }

    /// Close all file descriptors that have the `FD_CLOEXEC` flag set.
    ///
    /// Called during `exec` to implement close-on-exec semantics.  File
    /// descriptors without `FD_CLOEXEC` are preserved across the exec.
    pub fn close_on_exec(&mut self) {
        for slot in self.entries.iter_mut() {
            let should_close = slot
                .as_ref()
                .map(|e| e.fd_flags & FD_CLOEXEC != 0)
                .unwrap_or(false);
            if should_close {
                if let Some(entry) = slot.take() {
                    entry.node.close();
                }
            }
        }
    }
}

impl Default for FdTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::{VfsNode, VfsStat};
    use abi::errors::Errno;
    use alloc::sync::Arc;

    struct NullNode;
    impl VfsNode for NullNode {
        fn read(&self, _: u64, _: &mut [u8]) -> SysResult<usize> {
            Ok(0)
        }
        fn write(&self, _: u64, buf: &[u8]) -> SysResult<usize> {
            Ok(buf.len())
        }
        fn stat(&self) -> SysResult<VfsStat> {
            Ok(VfsStat {
                mode: VfsStat::S_IFCHR | 0o666,
                size: 0,
                ino: 1,
                ..Default::default()
            })
        }
    }

    fn null_node() -> Arc<dyn VfsNode> {
        Arc::new(NullNode)
    }

    #[test]
    fn test_open_allocates_from_0_when_empty() {
        let mut table = FdTable::new();
        let fd = table
            .open(null_node(), OpenFlags::read_only(), "/null".into())
            .unwrap();
        assert_eq!(fd, 0, "first fd in empty table should be 0");
    }

    #[test]
    fn test_open_skips_occupied_slots() {
        let mut table = FdTable::new();
        // Pre-populate slots 0-2 (simulate stdio setup).
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/in".into())
            .unwrap();
        table
            .insert_at(1, null_node(), OpenFlags::write_only(), "/out".into())
            .unwrap();
        table
            .insert_at(2, null_node(), OpenFlags::write_only(), "/err".into())
            .unwrap();
        let fd = table
            .open(null_node(), OpenFlags::read_only(), "/null".into())
            .unwrap();
        assert_eq!(fd, 3, "first non-stdio VFS fd should be 3");
    }

    #[test]
    fn test_open_sequential_fds() {
        let mut table = FdTable::new();
        // Pre-populate slots 0-2 (simulate stdio setup).
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/in".into())
            .unwrap();
        table
            .insert_at(1, null_node(), OpenFlags::write_only(), "/out".into())
            .unwrap();
        table
            .insert_at(2, null_node(), OpenFlags::write_only(), "/err".into())
            .unwrap();
        let fd1 = table
            .open(null_node(), OpenFlags::read_only(), "/f1".into())
            .unwrap();
        let fd2 = table
            .open(null_node(), OpenFlags::read_only(), "/f2".into())
            .unwrap();
        assert_eq!(fd1, 3);
        assert_eq!(fd2, 4);
    }

    #[test]
    fn test_get_unknown_fd_returns_ebadf() {
        let table = FdTable::new();
        assert!(matches!(table.get(10), Err(Errno::EBADF)));
    }

    #[test]
    fn test_close_frees_slot() {
        let mut table = FdTable::new();
        let fd = table
            .open(null_node(), OpenFlags::read_only(), "/null".into())
            .unwrap();
        table.close(fd).unwrap();
        assert!(matches!(table.get(fd), Err(Errno::EBADF)));
    }

    #[test]
    fn test_close_reuses_slot() {
        let mut table = FdTable::new();
        let fd1 = table
            .open(null_node(), OpenFlags::read_only(), "/f1".into())
            .unwrap();
        table.close(fd1).unwrap();
        let fd2 = table
            .open(null_node(), OpenFlags::read_only(), "/f2".into())
            .unwrap();
        // Slot 0 was freed, so it should be reused.
        assert_eq!(fd2, 0);
    }

    #[test]
    fn test_close_ebadf_for_not_open() {
        let mut table = FdTable::new();
        assert!(matches!(table.close(99), Err(Errno::EBADF)));
    }

    #[test]
    fn test_insert_at_populates_specific_slot() {
        let mut table = FdTable::new();
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/in".into())
            .unwrap();
        table
            .insert_at(1, null_node(), OpenFlags::write_only(), "/out".into())
            .unwrap();
        table
            .insert_at(2, null_node(), OpenFlags::write_only(), "/err".into())
            .unwrap();
        assert!(table.get(0).is_ok());
        assert!(table.get(1).is_ok());
        assert!(table.get(2).is_ok());
    }

    #[test]
    fn test_insert_at_rejects_occupied_slot() {
        let mut table = FdTable::new();
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/null".into())
            .unwrap();
        assert!(matches!(
            table.insert_at(0, null_node(), OpenFlags::read_only(), "/null".into()),
            Err(Errno::EBADF)
        ));
    }

    #[test]
    fn test_insert_at_rejects_out_of_range() {
        let mut table = FdTable::new();
        assert!(matches!(
            table.insert_at(
                MAX_FDS as u32,
                null_node(),
                OpenFlags::read_only(),
                "/null".into()
            ),
            Err(Errno::EBADF)
        ));
    }

    #[test]
    fn test_dup_clones_to_next_free() {
        let mut table = FdTable::new();
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/null".into())
            .unwrap();
        let new_fd = table.dup(0).unwrap();
        assert_eq!(new_fd, 1, "dup should use first free slot after 0");
        assert!(table.get(1).is_ok());
    }

    #[test]
    fn test_dup_ebadf_for_closed_fd() {
        let mut table = FdTable::new();
        assert!(matches!(table.dup(5), Err(Errno::EBADF)));
    }

    #[test]
    fn test_dup2_creates_alias() {
        let mut table = FdTable::new();
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/null".into())
            .unwrap();
        let result = table.dup2(0, 5).unwrap();
        assert_eq!(result, 5);
        assert!(table.get(5).is_ok());
        // Original still open.
        assert!(table.get(0).is_ok());
    }

    #[test]
    fn test_dup2_closes_existing_target() {
        let mut table = FdTable::new();
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/in".into())
            .unwrap();
        table
            .insert_at(1, null_node(), OpenFlags::write_only(), "/out".into())
            .unwrap();
        // dup2(0, 1) should close slot 1 and replace it with a dup of slot 0.
        table.dup2(0, 1).unwrap();
        assert!(table.get(1).is_ok());
    }

    #[test]
    fn test_dup2_same_fd_is_noop() {
        let mut table = FdTable::new();
        table
            .insert_at(3, null_node(), OpenFlags::read_only(), "/null".into())
            .unwrap();
        let result = table.dup2(3, 3).unwrap();
        assert_eq!(result, 3);
        assert!(table.get(3).is_ok());
    }

    #[test]
    fn test_dup2_ebadf_for_closed_old_fd() {
        let mut table = FdTable::new();
        assert!(matches!(table.dup2(99, 5), Err(Errno::EBADF)));
    }

    #[test]
    fn test_dup_shares_offset() {
        let mut table = FdTable::new();
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/null".into())
            .unwrap();
        let new_fd = table.dup(0).unwrap();
        // Advance the original fd's offset.
        *table.get(0).unwrap().offset.lock() = 42;
        // Duplicated fd should see the same offset.
        let dup_offset = *table.get(new_fd).unwrap().offset.lock();
        assert_eq!(dup_offset, 42, "dup should share file offset");
    }

    #[test]
    fn test_dup2_shares_offset() {
        let mut table = FdTable::new();
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/null".into())
            .unwrap();
        table.dup2(0, 5).unwrap();
        // Advance via fd 5.
        *table.get(5).unwrap().offset.lock() = 100;
        // fd 0 should see the same value.
        let orig_offset = *table.get(0).unwrap().offset.lock();
        assert_eq!(orig_offset, 100, "dup2 should share file offset");
    }

    #[test]
    fn test_dup_shares_status_flags() {
        let mut table = FdTable::new();
        table
            .insert_at(
                0,
                null_node(),
                OpenFlags(abi::syscall::vfs_flags::O_RDONLY),
                "/null".into(),
            )
            .unwrap();
        let new_fd = table.dup(0).unwrap();

        *table.get(0).unwrap().status_flags.lock() =
            OpenFlags(abi::syscall::vfs_flags::O_RDONLY | abi::syscall::vfs_flags::O_NONBLOCK);

        let dup_flags = *table.get(new_fd).unwrap().status_flags.lock();
        assert!(dup_flags.is_nonblock(), "dup should share status flags");
    }

    #[test]
    fn test_dup_clears_descriptor_flags() {
        let mut table = FdTable::new();
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/null".into())
            .unwrap();
        table.set_fd_flags(0, FD_CLOEXEC).unwrap();

        let new_fd = table.dup(0).unwrap();

        assert_eq!(table.get_fd_flags(0).unwrap(), FD_CLOEXEC);
        assert_eq!(
            table.get_fd_flags(new_fd).unwrap(),
            0,
            "dup should not inherit FD_CLOEXEC"
        );
    }

    #[test]
    fn test_emfile_when_full() {
        // Open MAX_FDS files and ensure EMFILE on the next open.
        let mut table = FdTable::new();
        let mut count = 0usize;
        loop {
            match table.open(null_node(), OpenFlags::read_only(), "/null".to_string()) {
                Ok(_) => count += 1,
                Err(Errno::EMFILE) => break,
                Err(e) => panic!("unexpected error {:?}", e),
            }
        }
        assert_eq!(count, MAX_FDS);
    }

    // ── close_on_exec tests ───────────────────────────────────────────────────

    /// FDs with FD_CLOEXEC set should be closed; others should survive.
    #[test]
    fn test_close_on_exec_closes_flagged_fds() {
        let mut table = FdTable::new();
        // fd 0: no flag → should survive exec
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/in".into())
            .unwrap();
        // fd 1: FD_CLOEXEC → should be closed on exec
        table
            .insert_at(1, null_node(), OpenFlags::write_only(), "/out".into())
            .unwrap();
        table.set_fd_flags(1, FD_CLOEXEC).unwrap();
        // fd 3: FD_CLOEXEC → should be closed on exec
        table
            .insert_at(3, null_node(), OpenFlags::read_only(), "/extra".into())
            .unwrap();
        table.set_fd_flags(3, FD_CLOEXEC).unwrap();

        table.close_on_exec();

        assert!(table.get(0).is_ok(), "fd 0 (no FD_CLOEXEC) should survive");
        assert!(
            matches!(table.get(1), Err(Errno::EBADF)),
            "fd 1 (FD_CLOEXEC) should be closed"
        );
        assert!(
            matches!(table.get(3), Err(Errno::EBADF)),
            "fd 3 (FD_CLOEXEC) should be closed"
        );
    }

    /// If no FDs are flagged, close_on_exec is a no-op.
    #[test]
    fn test_close_on_exec_preserves_unflagged_fds() {
        let mut table = FdTable::new();
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/in".into())
            .unwrap();
        table
            .insert_at(1, null_node(), OpenFlags::write_only(), "/out".into())
            .unwrap();

        table.close_on_exec();

        assert!(table.get(0).is_ok(), "fd 0 should survive");
        assert!(table.get(1).is_ok(), "fd 1 should survive");
    }

    /// An empty table is handled gracefully.
    #[test]
    fn test_close_on_exec_empty_table() {
        let mut table = FdTable::new();
        // Should not panic.
        table.close_on_exec();
    }

    /// After close_on_exec the closed slot can be reused.
    #[test]
    fn test_close_on_exec_slot_reuse() {
        let mut table = FdTable::new();
        table
            .insert_at(0, null_node(), OpenFlags::read_only(), "/f".into())
            .unwrap();
        table.set_fd_flags(0, FD_CLOEXEC).unwrap();

        table.close_on_exec();

        // Slot 0 is now free; the next open() should reuse it.
        let new_fd = table
            .open(null_node(), OpenFlags::read_only(), "/new".into())
            .unwrap();
        assert_eq!(new_fd, 0, "freed cloexec slot should be reusable");
    }
}
