# Filesystem Semantics Checklist

This document describes the current state of POSIX-like filesystem semantics in
Thing-OS, enumerates known gaps, and records the acceptance criteria required to
call each area "done".

> **Related**: `docs/posix-checklist.toml` is the machine-readable tracker for
> individual POSIX features.  This document provides the narrative rationale and
> per-operation detail.

---

## 1. `open / O_CREAT / O_EXCL / O_TRUNC`

| Behaviour | Status | Notes |
|-----------|--------|-------|
| Open existing file | ✅ | `vfs::mount::lookup` |
| Create new file (`O_CREAT`) | ✅ | Falls back to `VfsDriver::create` when lookup returns ENOENT |
| Exclusive create (`O_CREAT\|O_EXCL`) | ✅ | Returns `EEXIST` if the path already exists |
| Truncate on open (`O_TRUNC`) | ✅ | Calls `VfsNode::truncate(0)` on the existing node |
| Open non-existent without `O_CREAT` | ✅ | Returns `ENOENT` |
| `O_APPEND` mode | ✅ | `fd_flags::O_APPEND` tracked in `FdEntry`; write handlers advance offset to EOF before each write |
| `O_NONBLOCK` | ✅ | Tracked in flags; used by `poll` readiness probing |
| Path length validation (0 / >4096) | ✅ | `sys_fs_open` rejects with `EINVAL` |

---

## 2. `read / write / readv / writev`

| Behaviour | Status | Notes |
|-----------|--------|-------|
| Basic read/write | ✅ | |
| Scatter/gather (`readv` / `writev`) | ✅ | `SYS_FS_READV` / `SYS_FS_WRITEV` |
| Short read at EOF returns 0 | ✅ | `VfsNode::read` contract |
| Write advances the shared offset | ✅ | `Arc<Mutex<u64>>` offset in `FdEntry` |
| Append mode always writes at EOF | ✅ | offset snapped to file size before write |
| Invalid user pointer returns `EFAULT` | ✅ | `validate_user_range` |

---

## 3. `lseek` (seek)

| Behaviour | Status | Notes |
|-----------|--------|-------|
| `SEEK_SET` to positive offset | ✅ | |
| `SEEK_SET` with negative value → `EINVAL` | ✅ | Fixed: raw bits reinterpreted as `i64` |
| `SEEK_CUR` with positive delta | ✅ | |
| `SEEK_CUR` with negative delta | ✅ | Fixed: checked signed arithmetic |
| `SEEK_CUR` that would go before 0 → `EINVAL` | ✅ | |
| `SEEK_END` with 0 delta → returns file size | ✅ | |
| `SEEK_END` with negative delta | ✅ | Fixed: checked signed arithmetic |
| `SEEK_END` with positive delta (sparse hole) | ✅ | Offset may exceed file size |
| `SEEK_END` that would underflow → `EINVAL` | ✅ | |
| Unknown `whence` → `EINVAL` | ✅ | |

> **Previous bug (now fixed)**: `sys_fs_seek` treated the raw `offset` argument
> as unsigned (`usize`) for `SEEK_CUR` and `SEEK_END`.  A negative `i64`
> transmitted as `usize` was then cast to a very large `u64`, causing
> `saturating_add` to clamp at `u64::MAX`.  The fix reinterprets the raw bits
> as `i64` and uses checked signed arithmetic.

---

## 4. `stat / fstat / lstat`

| Behaviour | Status | Notes |
|-----------|--------|-------|
| `mode`, `size`, `ino` | ✅ | |
| `atime` / `mtime` / `ctime` | ✅ | `VfsStat` carries `{a,m,c}time_{sec,nsec}`; updated on write/truncate/mkdir |
| `nlink`, `uid`, `gid`, `rdev` | ✅ | `VfsStat` fields; hard-link count maintained |
| `blksize`, `blocks` derived | ✅ | `to_abi_stat()` derives `blksize=4096`, `blocks=ceil(size/512)` |
| `lstat` (no symlink follow) | ✅ | `SYS_FS_LSTAT` / `resolve_no_follow()` |
| Path length validation | ✅ | |

---

## 5. `readdir`

| Behaviour | Status | Notes |
|-----------|--------|-------|
| List entries via `SYS_FS_READDIR` | ✅ | `sys_fs_readdir` calls `VfsNode::readdir` |
| `.` and `..` entries | ⚠️ partial | Synthetic `.`/`..` are not currently injected by ramfs or devfs |
| Entry `d_type` field | ✅ | `DirEntry` carries `file_type` from `VfsStat::mode` |

**Gap**: `readdir` does not synthesize `.` (current directory) and `..`
(parent directory) entries.  Programs that depend on iterating these will see
them as absent.  Tracked as a known gap.

---

## 6. `mkdir`

| Behaviour | Status | Notes |
|-----------|--------|-------|
| Create new directory | ✅ | `SYS_FS_MKDIR` → `VfsDriver::mkdir` |
| `EEXIST` when final component already exists | ✅ | Fixed: `VfsDriver::mkdir` in ramfs now checks the last component |
| Intermediate directories created on demand | ✅ | Intentional ThingOS extension (like `mkdir -p`) |
| Parent must be a directory | ✅ | `insert_child` returns `ENOTDIR` for non-directory parents |
| Path length validation | ✅ | |

> **Previous gap (now fixed)**: `VfsDriver::mkdir` in ramfs used "mkdir -p"
> semantics and silently succeeded when the target directory already existed.
> It now returns `EEXIST` when the final path component already exists.

---

## 7. `unlink / rmdir`

| Behaviour | Status | Notes |
|-----------|--------|-------|
| Remove a regular file | ✅ | `SYS_FS_UNLINK` → `VfsDriver::unlink` |
| Remove a directory | ✅ | Unified unlink handles both (no separate `rmdir` syscall yet) |
| Non-empty directory removal | ⚠️ | Currently succeeds even if non-empty – **known gap** |
| Hard-link count decremented | ✅ | `nlink` updated in `VfsDriver::unlink` |
| Path not found → `ENOENT` | ✅ | |
| Parent not a directory → `ENOTDIR` | ✅ | |

**Gap**: There is no separate `SYS_FS_RMDIR` syscall. The current `unlink`
does not refuse to remove non-empty directories (`ENOTEMPTY` is unimplemented).
POSIX `unlink(2)` should return `EISDIR` when called on a directory; POSIX
`rmdir(2)` should return `ENOTEMPTY` for non-empty directories.  These will be
addressed when a distinct `rmdir` syscall is added.

---

## 8. `rename`

| Behaviour | Status | Notes |
|-----------|--------|-------|
| Same-directory rename | ✅ | |
| Cross-directory rename (same fs) | ✅ | Lock ordering by pointer to avoid deadlock |
| Cross-mount rename → `EXDEV` | ✅ | `mount::rename` checks same driver |
| Source not found → `ENOENT` | ✅ | |
| Overwrite existing destination | ✅ | Destination entry is replaced atomically within the lock |
| Rename directory onto non-empty dir | ⚠️ | Does not check `ENOTEMPTY` – known gap |

---

## 9. `chdir / getcwd`

| Behaviour | Status | Notes |
|-----------|--------|-------|
| `chdir` to an existing directory | ✅ | Updates `ProcessInfo.cwd` |
| `chdir` to a non-directory → `ENOTDIR` | ✅ | Checked via `stat().is_dir()` |
| `chdir` to non-existent path → `ENOENT` | ✅ | `mount::lookup` returns `ENOENT` |
| `chdir` with empty/overlong path → `EINVAL` | ✅ | Fixed: length guard added to `sys_fs_chdir` |
| `getcwd` returns current CWD | ✅ | `ProcessInfo.cwd` returned verbatim |
| Relative path resolution uses CWD | ✅ | `resolve_path()` prepends `cwd` for non-absolute paths |
| `.` and `..` in relative paths normalised | ✅ | `vfs::path::normalise()` |
| CWD is inherited by child processes | ✅ | `ProcessInfo` cloned at spawn |

> **Previous gap (now fixed)**: `sys_fs_chdir` was missing the `path_len == 0
> || path_len > 4096` guard present in other path-handling syscalls, which
> could cause the kernel to attempt a copy from a bogus user pointer for
> oversized path lengths.

---

## 10. `realpath`

| Behaviour | Status | Notes |
|-----------|--------|-------|
| Canonical absolute path from absolute input | ✅ | `sys_fs_realpath` → `vfs::path::normalise` |
| Canonical path from relative input | ✅ | `resolve_path` prepends CWD then normalises |
| Symlink expansion | ✅ | `vfs::path::resolve_at` expands up to 40 levels |
| `ELOOP` after too many symlinks | ✅ | |

---

## 11. Path length and component limits

| Behaviour | Status | Notes |
|-----------|--------|-------|
| Path longer than 4096 → `EINVAL` | ✅ | Checked in every path-accepting syscall |
| More than 64 path components → `ENAMETOOLONG` | ✅ | `vfs::path::normalise` enforces `MAX_COMPONENTS=64` |

---

## 12. Mount-provider consistency

| Mount point | Driver | open | read | write | readdir | mkdir | unlink | rename |
|-------------|--------|------|------|-------|---------|-------|--------|--------|
| `/`         | ramfs  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/tmp`      | ramfs  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/run`      | ramfs  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/services` | ramfs  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/dev`      | devfs  | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ |
| `/proc`     | procfs | ✅ | ✅ | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| `/boot`     | bootfs | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |

> devfs mkdir/unlink/rename are not applicable to the built-in device nodes but
> can be registered via `devfs::register`/`unregister`.  procfs is intentionally
> read-only.  bootfs is a static initramfs (read-only).

---

## Known Gaps and Tracked Issues

| # | Gap | Severity | Owner |
|---|-----|----------|-------|
| FS-1 | `readdir` does not emit `.` and `..` synthetic entries | Medium | kernel |
| FS-2 | `unlink` does not return `EISDIR`; no separate `rmdir` syscall | Medium | kernel |
| FS-3 | `unlink` / `rmdir` do not check for non-empty directories (`ENOTEMPTY`) | Medium | kernel |
| FS-4 | `rename` does not check `ENOTEMPTY` when replacing a non-empty directory | Low | kernel |
| FS-5 | `fcntl` `F_GETFL` / `F_SETFL` support is incomplete | Low | mixed |
| FS-6 | No `O_DIRECTORY` flag enforcement on `open` | Low | kernel |
| FS-7 | `devfs` does not support user-visible `mkdir` / `rename` / `unlink` on virtual device directories | Low | kernel |
| FS-8 | `procfs` is a stub; most `/proc/*` files return empty or ENOENT | Medium | kernel |
| FS-9 | No `faccessat` / `openat` / `mkdirat` AT_FDCWD variant syscalls | Low | kernel |
| FS-10 | POSIX signals / `EINTR` not yet propagated through blocking VFS calls | High | kernel |

---

## Regression Test Coverage

Kernel unit tests (run with `PCI_IDS_MODE=stub cargo test -p kernel --lib`):

- `vfs::path::tests` — `normalise()` edge cases, `.` / `..` collapsing, `ENAMETOOLONG`
- `vfs::ramfs::tests` — create, read, write, seek, mkdir (including EEXIST), unlink, rename, truncate, symlinks, hard links, stat timestamps
- `vfs::devfs::tests` — null/zero/console read/write/stat, directory listing
- `vfs::union::tests` — layer shadowing, fallthrough, error propagation
- `syscall::handlers::vfs::tests` — `sys_fs_seek` signed-offset semantics (all three whence values), `sys_fs_getcwd`, `sys_fs_chdir` input validation, `resolve_path` relative-path resolution, `sys_fs_poll` readiness

Userspace integration tests (`userspace/tests/test_fs`):

- `fs_create_write_read` — basic file I/O round-trip
- `fs_seek` — SEEK_SET / SEEK_CUR / SEEK_END (positive and negative)
- `fs_readdir` — directory listing
- `fs_metadata` — stat for files and directories
- `fs_rename_remove` — rename and unlink
- `fs_create_new` — O_CREAT | O_EXCL
- `fs_truncate` — ftruncate
- `fs_chdir_relative_open` — chdir then open by relative path, getcwd verification
- `fs_getcwd_roundtrip` — chdir → getcwd consistency
- `fs_mkdir_eexist` — mkdir on existing directory returns EEXIST
- `fs_open_enoent` — open without O_CREAT on missing path returns NotFound
