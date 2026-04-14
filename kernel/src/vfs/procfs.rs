//! procfs — process information filesystem mounted at `/proc`.
//!
//! Provides a read-only view of running processes and system state.
//!
//! # Paths exposed
//!
//! | Path                             | Contents |
//! |----------------------------------|----------|
//! | `/proc/version`                  | Kernel version string |
//! | `/proc/mounts`                   | Active mount table (text) |
//! | `/proc/meminfo`                  | Heap memory statistics |
//! | `/proc/cpuinfo`                  | CPU model and frequency |
//! | `/proc/uptime`                   | Seconds since boot |
//! | `/proc/self`                     | Directory for the calling process |
//! | `/proc/self/exe`                 | Symlink to calling process's executable |
//! | `/proc/<pid>/status`             | Process state, name, ppid |
//! | `/proc/<pid>/cmdline`            | argv as null-delimited bytes |
//! | `/proc/<pid>/exe`                | Symlink to the process's executable |
//! | `/proc/<pid>/fd/`                | Directory of open fd targets |
//! | `/proc/<pid>/task/`              | Directory of threads in the process |
//! | `/proc/<pid>/task/<tid>/name`    | Thread's human-readable name |
//! | `/proc/<pid>/task_state`         | Canonical `thingos::task::TaskState` (Phase 1) |
//! | `/proc/<pid>/job_state`          | Canonical `thingos::job::JobState` (Phase 2) |
//! | `/proc/<pid>/job_exit`           | Canonical `thingos::job::JobExit` — state + code (Phase 3) |
//! | `/proc/<pid>/job_wait`           | Canonical `thingos::job::JobWaitResult` — non-blocking poll (Phase 3) |
//! | `/proc/<pid>/group_kind`         | Canonical `thingos::group::GroupKind` — coordination role (Phase 4)     |
//! | `/proc/<pid>/authority`          | Canonical `thingos::authority::Authority` — permission context (Phase 7) |
//! | `/proc/<pid>/place`              | Canonical `thingos::place::Place` — world/visibility context (Phase 8)  |

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use abi::errors::{Errno, SysResult};

use super::{VfsDriver, VfsNode, VfsStat};

// ── ProcFs driver ─────────────────────────────────────────────────────────────

/// The process filesystem driver.  Mounted at `/proc` by `vfs::init`.
pub struct ProcFs;

impl ProcFs {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ProcFs {
    fn default() -> Self {
        Self::new()
    }
}

impl VfsDriver for ProcFs {
    fn lookup(&self, path: &str) -> SysResult<Arc<dyn VfsNode>> {
        match path {
            "" => Ok(Arc::new(ProcDirNode)),
            "version" => Ok(Arc::new(StaticTextNode::new(b"Thing-OS v0.2\n", 2))),
            "mounts" => Ok(Arc::new(MountsNode)),
            "meminfo" => Ok(Arc::new(MemInfoNode)),
            "cpuinfo" => Ok(Arc::new(CpuInfoNode)),
            "uptime" => Ok(Arc::new(UptimeNode)),
            // /proc/ipc — IPC diagnostics directory
            "ipc" => Ok(Arc::new(IpcDirNode)),
            "ipc/channels" => Ok(Arc::new(IpcDiagNode::channels())),
            "ipc/pipes" => Ok(Arc::new(IpcDiagNode::pipes())),
            "ipc/vfs_rpc" => Ok(Arc::new(IpcDiagNode::vfs_rpc())),
            // /proc/self — virtual directory for the calling process
            "self" => Ok(Arc::new(ProcSelfDirNode)),
            // /proc/self/exe — symlink to the current process's executable
            "self/exe" => Ok(Arc::new(ProcSelfExeNode)),
            _ => {
                // Try to match /proc/<pid>/... paths.
                // `path` is already relative to the mount point, so it looks
                // like "42/status", "42/cmdline", "42", etc.
                let mut parts = path.splitn(2, '/');
                let pid_str = parts.next().unwrap_or("");
                let rest = parts.next().unwrap_or("");

                if let Ok(pid) = pid_str.parse::<u32>() {
                    return lookup_pid(pid, rest);
                }
                Err(Errno::ENOENT)
            }
        }
    }
}

/// Look up a node inside a per-process `/proc/<pid>/` directory.
fn lookup_pid(pid: u32, rest: &str) -> SysResult<Arc<dyn VfsNode>> {
    // For paths under "task/..." we delegate immediately without needing the
    // process snapshot (which is only used by status/cmdline/exe).
    if rest == "task" || rest.starts_with("task/") {
        let tid_and_rest = rest.strip_prefix("task/").unwrap_or("");
        return lookup_pid_task(pid, tid_and_rest);
    }

    let snap = process_snapshot(pid).ok_or(Errno::ENOENT)?;

    match rest {
        // /proc/<pid> — the per-process directory itself
        "" => Ok(Arc::new(ProcPidDirNode { pid })),
        "status" => {
            let state_name = match snap.state {
                crate::task::TaskState::Runnable => "R",
                crate::task::TaskState::Running => "R",
                crate::task::TaskState::Blocked => "S",
                crate::task::TaskState::Dead => "Z",
            };
            let text = alloc::format!(
                "Name:\t{}\nState:\t{}\nPid:\t{}\nPPid:\t{}\n",
                snap.name,
                state_name,
                snap.pid,
                snap.ppid,
            );
            Ok(Arc::new(DynamicTextNode::new(text.into_bytes(), 300 + pid as u64 * 10 + 1)))
        }
        "cmdline" => {
            // Standard Linux /proc/<pid>/cmdline format: each argument is
            // followed by a NUL byte (including the last), so the full content
            // is "arg0\0arg1\0arg2\0".
            let mut data: Vec<u8> = Vec::new();
            for arg in snap.argv.iter() {
                data.extend_from_slice(arg);
                data.push(0);
            }
            Ok(Arc::new(DynamicTextNode::new(data, 300 + pid as u64 * 10 + 2)))
        }
        "fd" => Ok(Arc::new(ProcPidFdDirNode { pid })),
        "exe" => {
            // /proc/<pid>/exe — symlink to the process's executable path.
            Ok(Arc::new(ProcPidExeNode { exec_path: snap.exec_path.clone() }))
        }
        // /proc/<pid>/task_state — canonical thingos::task::TaskState label.
        //
        // Bridges the current kernel ThreadState into the schema-generated
        // TaskState via `kernel::task::bridge`.  This is the first public
        // surface for the Task ontology (Phase 1).
        "task_state" => {
            let task_state = crate::task::bridge::task_state_from_thread(snap.state);
            let text = alloc::format!("{}\n", task_state.as_str());
            Ok(Arc::new(DynamicTextNode::new(text.into_bytes(), 300 + pid as u64 * 10 + 5)))
        }
        // /proc/<pid>/job_state — canonical thingos::job::JobState label.
        //
        // Bridges the current kernel Process/Thread lifecycle into the
        // schema-generated JobState via `kernel::job::bridge`.  This is the
        // first public surface for the Job ontology (Phase 2).
        "job_state" => {
            let job_state = crate::job::bridge::job_state_from_snapshot(&snap);
            let text = alloc::format!("{}\n", job_state.as_str());
            Ok(Arc::new(DynamicTextNode::new(text.into_bytes(), 300 + pid as u64 * 10 + 6)))
        }
        // /proc/<pid>/job_exit — canonical thingos::job::JobExit (Phase 3).
        //
        // Reports the exit state and code in canonical Job terms, bridged from
        // the current Process/Thread model via `kernel::job::bridge`.
        // For live processes `code` is reported as `-`.
        "job_exit" => {
            let job_exit = crate::job::bridge::job_exit_from_snapshot(&snap);
            let text = job_exit.as_text();
            Ok(Arc::new(DynamicTextNode::new(text.into_bytes(), 300 + pid as u64 * 10 + 7)))
        }
        // /proc/<pid>/job_wait — canonical thingos::job::JobWaitResult (Phase 3).
        //
        // Non-blocking poll of the job's wait result.  Reinterprets the
        // current `poll_task_exit` output through the canonical Job vocabulary
        // via `kernel::job::bridge::job_wait_result_from_poll`.
        "job_wait" => {
            // poll_task_exit_current takes a TaskId (tid).  For the process
            // leader snap.tid == snap.pid as u64.
            let poll = unsafe { crate::sched::poll_task_exit_current(snap.tid) }
                .map_err(|_| Errno::ENOENT)?;
            let wait_result = crate::job::bridge::job_wait_result_from_poll(poll);
            let text = wait_result.as_text();
            Ok(Arc::new(DynamicTextNode::new(text.into_bytes(), 300 + pid as u64 * 10 + 8)))
        }
        // /proc/<pid>/group_kind — canonical thingos::group::GroupKind (Phase 4).
        //
        // Reports the coordination role of this process's group in canonical
        // Group terms, bridged from the current session_leader field via
        // `kernel::group::bridge`.  This is the first public surface for the
        // Group ontology (Phase 4).
        "group_kind" => {
            let group = crate::group::bridge::group_from_snapshot(&snap);
            let text = group.as_text();
            Ok(Arc::new(DynamicTextNode::new(text.into_bytes(), 300 + pid as u64 * 10 + 9)))
        }
        // /proc/<pid>/authority — canonical thingos::authority::Authority (Phase 7).
        //
        // Reports the active permission context in canonical Authority terms,
        // bridged from the current Process-shaped credential state via
        // `kernel::authority::bridge`.  This is the first public surface for
        // the Authority ontology (Phase 7).
        //
        // In Phase 7 the authority name is derived from the process name and
        // capabilities is always empty (the current `Process` carries no
        // explicit capability mask).  Future phases will populate capabilities
        // once uid/gid-like fields or a capability mask are introduced into
        // the Process struct.
        "authority" => {
            let authority = crate::authority::bridge::authority_from_snapshot(&snap);
            let text = authority.as_text();
            Ok(Arc::new(DynamicTextNode::new(text.into_bytes(), 300 + pid as u64 * 10 + 10)))
        }
        // /proc/<pid>/place — canonical thingos::place::Place (Phase 8).
        //
        // Reports the execution world-context (cwd, namespace, root) in
        // canonical Place terms, bridged from the current Process-shaped
        // cwd/namespace state via `kernel::place::bridge`.  This is the first
        // public surface for the Place ontology (Phase 8).
        //
        // In Phase 8:
        // * `cwd` is derived from Process::cwd.
        // * `namespace` is always "global" (NamespaceRef is a unit struct).
        // * `root` is always "/" (no per-process chroot yet).
        //
        // Note: terminal/UI/console attachment is NOT reported here.  That
        // belongs to Presence, which has not yet been introduced as a live
        // execution concept.  This path answers "in what world?", not
        // "who is present?".
        "place" => {
            let place = crate::place::bridge::place_from_snapshot(&snap);
            let text = place.as_text();
            Ok(Arc::new(DynamicTextNode::new(text.into_bytes(), 300 + pid as u64 * 10 + 11)))
        }
        _ => Err(Errno::ENOENT),
    }
}
///
/// `tid_and_rest` is everything after `"task/"`, e.g. `""` (the directory
/// itself), `"100"` (per-thread directory), or `"100/name"` (thread name).
fn lookup_pid_task(pid: u32, tid_and_rest: &str) -> SysResult<Arc<dyn VfsNode>> {
    // /proc/<pid>/task — directory listing all TIDs.
    if tid_and_rest.is_empty() {
        return Ok(Arc::new(ProcPidTaskDirNode { pid }));
    }

    let mut parts = tid_and_rest.splitn(2, '/');
    let tid_str = parts.next().unwrap_or("");
    let file = parts.next().unwrap_or("");

    let tid: u64 = tid_str.parse().map_err(|_| Errno::ENOENT)?;

    // Find the thread snapshot with the matching pid and tid.
    let procs = crate::sched::list_processes_current();
    let thread = procs.iter().find(|s| s.pid == pid && s.tid == tid).ok_or(Errno::ENOENT)?;

    match file {
        // /proc/<pid>/task/<tid> — per-thread directory.
        "" => Ok(Arc::new(ProcPidTaskTidDirNode { pid, tid })),
        "name" => {
            // Thread name, newline-terminated for compatibility with Linux.
            let mut text = thread.name.clone();
            text.push('\n');
            // Inode: top nibble 0xB, next 32 bits = pid, bottom 28 bits = tid.
            // This avoids collisions for the expected TID range (< 2^28).
            let ino = 0xB000_0000_0000_0000u64 | ((pid as u64) << 28) | (tid & 0x0FFF_FFFF);
            Ok(Arc::new(DynamicTextNode::new(text.into_bytes(), ino)))
        }
        "task_state" => {
            // Canonical Task-shaped state derived via the bridge layer.
            //
            // Maps the kernel's internal ThreadState to the public thingos.task
            // vocabulary defined in tools/kindc/kinds/task.kind, using the
            // explicit bridge in crate::task::bridge rather than ad-hoc
            // conversion.  This is the first real code path that exposes the
            // new ontology at a system boundary.
            let task = crate::task::bridge::thread_state_to_task(thread.state);
            let state_name = match task.state {
                thingos::task::TaskState::New => "new",
                thingos::task::TaskState::Ready => "ready",
                thingos::task::TaskState::Running => "running",
                thingos::task::TaskState::Blocked => "blocked",
                thingos::task::TaskState::Exited => "exited",
            };
            let text = alloc::format!("state: {}\n", state_name);
            let ino = 0xC000_0000_0000_0000u64 | ((pid as u64) << 28) | (tid & 0x0FFF_FFFF);
            Ok(Arc::new(DynamicTextNode::new(text.into_bytes(), ino)))
        }
        _ => Err(Errno::ENOENT),
    }
}

fn process_snapshot(pid: u32) -> Option<crate::sched::ProcessSnapshot> {
    let procs = crate::sched::list_processes_current();
    procs
        .iter()
        .find(|p| p.pid == pid && p.tid == pid as u64)
        .cloned()
        .or_else(|| procs.into_iter().find(|p| p.pid == pid))
}

fn process_ids() -> Vec<u32> {
    let mut seen = BTreeSet::new();
    let mut pids = Vec::new();
    for snap in crate::sched::list_processes_current() {
        if seen.insert(snap.pid) {
            pids.push(snap.pid);
        }
    }
    pids
}

// ── /proc root directory ──────────────────────────────────────────────────────

struct ProcDirNode;

impl VfsNode for ProcDirNode {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat { mode: VfsStat::S_IFDIR | 0o555, size: 0, ino: 200, ..Default::default() })
    }
    fn readdir(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let mut names = alloc::vec![
            String::from("version"),
            String::from("mounts"),
            String::from("meminfo"),
            String::from("cpuinfo"),
            String::from("uptime"),
            String::from("ipc"),
            String::from("self"),
        ];
        for pid in process_ids() {
            names.push(alloc::format!("{}", pid));
        }
        super::write_readdir_entries(names.iter().map(|s: &String| s.as_str()), offset, buf)
    }
}

// ── /proc/<pid>/ directory ────────────────────────────────────────────────────

struct ProcPidDirNode {
    pid: u32,
}

impl VfsNode for ProcPidDirNode {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFDIR | 0o555,
            size: 0,
            ino: 300 + self.pid as u64 * 10,
            ..Default::default()
        })
    }
    fn readdir(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        // Legacy procfs entries (transitional internal model):
        //   status, cmdline, fd, exe, task
        // Canonical schema entries (Phase 1–8):
        //   task_state, job_state, job_exit, job_wait, group_kind, authority, place
        let entries = [
            "status",
            "cmdline",
            "fd",
            "exe",
            "task",
            "task_state",
            "job_state",
            "job_exit",
            "job_wait",
            "group_kind",
            "authority",
            "place",
        ];
        super::write_readdir_entries(entries.into_iter(), offset, buf)
    }
}

// ── /proc/<pid>/fd/ directory ─────────────────────────────────────────────────

struct ProcPidFdDirNode {
    #[allow(dead_code)]
    pid: u32,
}

impl VfsNode for ProcPidFdDirNode {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFDIR | 0o555,
            size: 0,
            ino: 300 + self.pid as u64 * 10 + 3,
            ..Default::default()
        })
    }
    fn readdir(&self, _offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        // Stub: empty directory.
        let _ = buf;
        Ok(0)
    }
}

// ── /proc/<pid>/task/ directory ───────────────────────────────────────────────

struct ProcPidTaskDirNode {
    pid: u32,
}

impl VfsNode for ProcPidTaskDirNode {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFDIR | 0o555,
            size: 0,
            // Inode: top nibble 0x9, bottom 32 bits = pid.
            ino: 0x9000_0000_0000_0000u64 | self.pid as u64,
            ..Default::default()
        })
    }
    fn readdir(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let procs = crate::sched::list_processes_current();
        let tids: Vec<String> = procs
            .iter()
            .filter(|s| s.pid == self.pid)
            .map(|s| alloc::format!("{}", s.tid))
            .collect();
        super::write_readdir_entries(tids.iter().map(|s: &String| s.as_str()), offset, buf)
    }
}

// ── /proc/<pid>/task/<tid>/ directory ─────────────────────────────────────────

struct ProcPidTaskTidDirNode {
    #[allow(dead_code)]
    pid: u32,
    #[allow(dead_code)]
    tid: u64,
}

impl VfsNode for ProcPidTaskTidDirNode {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFDIR | 0o555,
            size: 0,
            // Inode: top nibble 0xA, next 32 bits = pid, bottom 28 bits = tid.
            ino: 0xA000_0000_0000_0000u64 | ((self.pid as u64) << 28) | (self.tid & 0x0FFF_FFFF),
            ..Default::default()
        })
    }
    fn readdir(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let entries = ["name", "task_state"];
        super::write_readdir_entries(entries.into_iter(), offset, buf)
    }
}

// ── /proc/self — virtual directory for the calling process ───────────────────

struct ProcSelfDirNode;

impl VfsNode for ProcSelfDirNode {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat { mode: VfsStat::S_IFDIR | 0o555, size: 0, ino: 210, ..Default::default() })
    }
    fn readdir(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let entries = ["exe"];
        super::write_readdir_entries(entries.into_iter(), offset, buf)
    }
}

// ── /proc/self/exe — symlink to the current process's executable ──────────────

/// A symlink node that resolves to the calling process's executable path.
///
/// Reading returns the path; `readlink` returns it directly for VFS consumers.
struct ProcSelfExeNode;

impl VfsNode for ProcSelfExeNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let target = self.readlink()?;
        let data = target.as_bytes();
        let off = offset as usize;
        if off >= data.len() {
            return Ok(0);
        }
        let n = (data.len() - off).min(buf.len());
        buf[..n].copy_from_slice(&data[off..off + n]);
        Ok(n)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EROFS)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat { mode: VfsStat::S_IFLNK | 0o777, size: 0, ino: 211, ..Default::default() })
    }
    fn readlink(&self) -> SysResult<String> {
        let pinfo = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
        let path = pinfo.lock().exec_path.clone();
        if path.is_empty() { Err(Errno::ENOENT) } else { Ok(path) }
    }
}

// ── /proc/<pid>/exe — symlink to a specific process's executable ──────────────

/// A symlink node that resolves to a given process's executable path.
struct ProcPidExeNode {
    exec_path: String,
}

impl VfsNode for ProcPidExeNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let target = self.readlink()?;
        let data = target.as_bytes();
        let off = offset as usize;
        if off >= data.len() {
            return Ok(0);
        }
        let n = (data.len() - off).min(buf.len());
        buf[..n].copy_from_slice(&data[off..off + n]);
        Ok(n)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EROFS)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFLNK | 0o777,
            size: 0,
            ino: 0, // dynamic; caller doesn't rely on stable ino for exe nodes
            ..Default::default()
        })
    }
    fn readlink(&self) -> SysResult<String> {
        if self.exec_path.is_empty() { Err(Errno::ENOENT) } else { Ok(self.exec_path.clone()) }
    }
}

// ── Static text node ──────────────────────────────────────────────────────────

/// Returns a fixed byte slice on read.
struct StaticTextNode {
    data: &'static [u8],
    ino: u64,
}

impl StaticTextNode {
    const fn new(data: &'static [u8], ino: u64) -> Self {
        Self { data, ino }
    }
}

impl VfsNode for StaticTextNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let off = offset as usize;
        if off >= self.data.len() {
            return Ok(0);
        }
        let avail = &self.data[off..];
        let n = avail.len().min(buf.len());
        buf[..n].copy_from_slice(&avail[..n]);
        Ok(n)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EROFS)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFREG | 0o444,
            size: self.data.len() as u64,
            ino: self.ino,
            ..Default::default()
        })
    }
}

// ── /proc/mounts ─────────────────────────────────────────────────────────────

/// A dynamic node that renders the current mount table as text.
///
/// The output mirrors a simplified `/proc/mounts` format:
/// ```text
/// <mount_point> ramfs rw 0 0
/// ```
struct MountsNode;

impl VfsNode for MountsNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        // Build the mounts listing dynamically from the global mount table.
        let listing = super::mount::mounts_text();
        let data = listing.as_bytes();
        let off = offset as usize;
        if off >= data.len() {
            return Ok(0);
        }
        let avail = &data[off..];
        let n = avail.len().min(buf.len());
        buf[..n].copy_from_slice(&avail[..n]);
        Ok(n)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EROFS)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFREG | 0o444,
            size: 0, // dynamic — size not known until read
            ino: 201,
            ..Default::default()
        })
    }
}

// ── Dynamic text node ─────────────────────────────────────────────────────────

/// Returns an owned byte vector on read.  Used for dynamically-generated
/// per-process text nodes such as `/proc/<pid>/status`.
struct DynamicTextNode {
    data: Vec<u8>,
    ino: u64,
}

impl DynamicTextNode {
    fn new(data: Vec<u8>, ino: u64) -> Self {
        Self { data, ino }
    }
}

impl VfsNode for DynamicTextNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let off = offset as usize;
        if off >= self.data.len() {
            return Ok(0);
        }
        let avail = &self.data[off..];
        let n = avail.len().min(buf.len());
        buf[..n].copy_from_slice(&avail[..n]);
        Ok(n)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EROFS)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFREG | 0o444,
            size: self.data.len() as u64,
            ino: self.ino,
            ..Default::default()
        })
    }
}

// ── /proc/meminfo ─────────────────────────────────────────────────────────────

/// Reports kernel heap statistics in a simplified `/proc/meminfo` format.
struct MemInfoNode;

impl VfsNode for MemInfoNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        // We report the static heap reservation size; detailed used/free
        // accounting is not yet tracked in the global allocator.
        let total_kb = (crate::memory::layout::KHEAP_SIZE / 1024) as u64;
        let text = alloc::format!("MemTotal:    {:8} kB\nMemFree:     {:8} kB\n", total_kb, 0u64,);
        let data = text.as_bytes();
        let off = offset as usize;
        if off >= data.len() {
            return Ok(0);
        }
        let n = (data.len() - off).min(buf.len());
        buf[..n].copy_from_slice(&data[off..off + n]);
        Ok(n)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EROFS)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat { mode: VfsStat::S_IFREG | 0o444, size: 0, ino: 202, ..Default::default() })
    }
}

// ── /proc/cpuinfo ─────────────────────────────────────────────────────────────

/// Reports a minimal CPU description.
struct CpuInfoNode;

impl VfsNode for CpuInfoNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let text = b"model name\t: Thing-OS virtual CPU\nprocessor\t: 0\n";
        let off = offset as usize;
        if off >= text.len() {
            return Ok(0);
        }
        let n = (text.len() - off).min(buf.len());
        buf[..n].copy_from_slice(&text[off..off + n]);
        Ok(n)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EROFS)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat { mode: VfsStat::S_IFREG | 0o444, size: 0, ino: 203, ..Default::default() })
    }
}

// ── /proc/uptime ──────────────────────────────────────────────────────────────

/// Returns seconds since boot as a decimal string.
struct UptimeNode;

impl VfsNode for UptimeNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let secs = uptime_secs();
        let text = alloc::format!("{}.00 {}.00\n", secs, secs);
        let data = text.as_bytes();
        let off = offset as usize;
        if off >= data.len() {
            return Ok(0);
        }
        let n = (data.len() - off).min(buf.len());
        buf[..n].copy_from_slice(&data[off..off + n]);
        Ok(n)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EROFS)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat { mode: VfsStat::S_IFREG | 0o444, size: 0, ino: 204, ..Default::default() })
    }
}

/// Return the number of seconds elapsed since boot using the runtime's
/// monotonic clock.  Returns 0 in test environments where the runtime hook
/// is not installed.
fn uptime_secs() -> u64 {
    // `runtime_base()` panics in test builds if the hook is not set up;
    // guard with a cfg flag so unit tests still pass.
    #[cfg(not(test))]
    {
        let rt = crate::runtime_base();
        let ticks = rt.mono_ticks();
        let freq = rt.mono_freq_hz();
        if freq == 0 {
            return 0;
        }
        ticks / freq
    }
    #[cfg(test)]
    {
        0
    }
}

// ── /proc/ipc/ directory ──────────────────────────────────────────────────────

/// Directory node for `/proc/ipc`.
struct IpcDirNode;

impl VfsNode for IpcDirNode {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat { mode: VfsStat::S_IFDIR | 0o555, size: 0, ino: 500, ..Default::default() })
    }
    fn readdir(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let entries = ["channels", "pipes", "vfs_rpc"];
        super::write_readdir_entries(entries.into_iter(), offset, buf)
    }
}

/// Dynamic text node that renders IPC diagnostic counters on demand.
struct IpcDiagNode {
    kind: IpcDiagKind,
    ino: u64,
}

enum IpcDiagKind {
    Channels,
    Pipes,
    VfsRpc,
}

impl IpcDiagNode {
    fn channels() -> Self {
        Self { kind: IpcDiagKind::Channels, ino: 501 }
    }
    fn pipes() -> Self {
        Self { kind: IpcDiagKind::Pipes, ino: 502 }
    }
    fn vfs_rpc() -> Self {
        Self { kind: IpcDiagKind::VfsRpc, ino: 503 }
    }

    fn render(&self) -> alloc::string::String {
        match self.kind {
            IpcDiagKind::Channels => crate::ipc::diag::channels_text(),
            IpcDiagKind::Pipes => crate::ipc::diag::pipes_text(),
            IpcDiagKind::VfsRpc => crate::ipc::diag::vfs_rpc_text(),
        }
    }
}

impl VfsNode for IpcDiagNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let text = self.render();
        let data = text.as_bytes();
        let off = offset as usize;
        if off >= data.len() {
            return Ok(0);
        }
        let n = (data.len() - off).min(buf.len());
        buf[..n].copy_from_slice(&data[off..off + n]);
        Ok(n)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EROFS)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFREG | 0o444,
            size: 0, // dynamic
            ino: self.ino,
            ..Default::default()
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(path: &str) -> SysResult<Arc<dyn VfsNode>> {
        ProcFs::new().lookup(path)
    }

    #[test]
    fn test_lookup_root_is_dir() {
        let node = lookup("").unwrap();
        let stat = node.stat().unwrap();
        assert!(stat.is_dir());
    }

    #[test]
    fn test_lookup_version() {
        let node = lookup("version").unwrap();
        let mut buf = [0u8; 64];
        let n = node.read(0, &mut buf).unwrap();
        assert!(n > 0);
        assert!(core::str::from_utf8(&buf[..n]).unwrap().contains("Thing-OS"));
    }

    #[test]
    fn test_version_is_readonly() {
        let node = lookup("version").unwrap();
        assert!(matches!(node.write(0, b"hack"), Err(Errno::EROFS)));
    }

    #[test]
    fn test_lookup_unknown_returns_enoent() {
        assert!(matches!(lookup("doesnotexist"), Err(Errno::ENOENT)));
    }

    #[test]
    fn test_readdir_root_lists_entries() {
        let node = lookup("").unwrap();
        let mut buf = [0u8; 256];
        let n = node.readdir(0, &mut buf).unwrap();
        assert!(n > 0);
        let s = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(s.contains("version"));
        assert!(s.contains("mounts"));
        assert!(s.contains("meminfo"));
        assert!(s.contains("cpuinfo"));
        assert!(s.contains("uptime"));
        assert!(s.contains("ipc"));
    }

    #[test]
    fn test_lookup_meminfo() {
        let node = lookup("meminfo").unwrap();
        let mut buf = [0u8; 128];
        let n = node.read(0, &mut buf).unwrap();
        assert!(n > 0);
        let s = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(s.contains("MemTotal"));
    }

    #[test]
    fn test_lookup_cpuinfo() {
        let node = lookup("cpuinfo").unwrap();
        let mut buf = [0u8; 128];
        let n = node.read(0, &mut buf).unwrap();
        assert!(n > 0);
        let s = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(s.contains("processor"));
    }

    #[test]
    fn test_lookup_uptime() {
        let node = lookup("uptime").unwrap();
        let mut buf = [0u8; 64];
        let n = node.read(0, &mut buf).unwrap();
        assert!(n > 0);
    }

    #[test]
    fn test_lookup_pid_enoent_when_no_processes() {
        // In test environment there are no real processes so any PID should
        // return ENOENT.
        assert!(matches!(lookup("1/status"), Err(Errno::ENOENT)));
    }

    #[test]
    fn test_readdir_root_lists_ipc() {
        let node = lookup("").unwrap();
        let mut buf = [0u8; 256];
        let n = node.readdir(0, &mut buf).unwrap();
        let s = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(s.contains("ipc"));
    }

    #[test]
    fn test_lookup_ipc_dir_is_dir() {
        let node = lookup("ipc").unwrap();
        let stat = node.stat().unwrap();
        assert!(stat.is_dir());
    }

    #[test]
    fn test_lookup_ipc_channels() {
        let node = lookup("ipc/channels").unwrap();
        let mut buf = [0u8; 256];
        let n = node.read(0, &mut buf).unwrap();
        assert!(n > 0);
        let s = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(s.contains("sends:"));
    }

    #[test]
    fn test_lookup_ipc_pipes() {
        let node = lookup("ipc/pipes").unwrap();
        let mut buf = [0u8; 256];
        let n = node.read(0, &mut buf).unwrap();
        assert!(n > 0);
        let s = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(s.contains("writes:"));
    }

    #[test]
    fn test_lookup_ipc_vfs_rpc() {
        let node = lookup("ipc/vfs_rpc").unwrap();
        let mut buf = [0u8; 256];
        let n = node.read(0, &mut buf).unwrap();
        assert!(n > 0);
        let s = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(s.contains("requests:"));
    }

    #[test]
    fn test_ipc_diag_nodes_are_readonly() {
        for path in &["ipc/channels", "ipc/pipes", "ipc/vfs_rpc"] {
            let node = lookup(path).unwrap();
            assert!(matches!(node.write(0, b"x"), Err(Errno::EROFS)));
        }
    }
}
