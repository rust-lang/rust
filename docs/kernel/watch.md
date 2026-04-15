# VFS Watch System

The VFS watch system provides a mechanism for userspace to monitor changes to the filesystem namespace (creation, deletion, moves) and file content (writes).

## Architecture

The system is built on three core components:

1.  **EventQueue**: A ring buffer of `WatchEvent` structures. Each `Watch` object has its own queue.
2.  **Watch**: A `VfsNode` implementation that acts as a handle to an event stream. It can be wait-many'd using `WaitKind::Fd`.
3.  **Registry**: A global mapping between `VfsNode` identities and subscribing `Watch` objects.

## Syscalls

### `sys_watch_fd(fd, mask, flags)`
Registers a watch on an existing open file descriptor. Returns a new `watch_fd`.

### `sys_watch_path(path, mask, flags)`
Registers a watch on a path. Returns a new `watch_fd`.

## Event Emission

Events are emitted from canonical VFS mutation points in the kernel:
- `sys_fs_write` -> `MODIFY`
- `sys_fs_unlink` -> `REMOVE`
- `sys_fs_mkdir` -> `CREATE`
- `sys_fs_rename` -> `MOVE_FROM` / `MOVE_TO`

## Rename Pair Linking

When a file is renamed, two events are generated: `MOVE_FROM` and `MOVE_TO`. These are linked by a 32-bit `cookie` value. If the rename occurs within the same directory, both events are delivered to that directory's watchers.

## Queue Overflow

If a watch's queue fills up, an `OVERFLOW` event is pushed, and subsequent events are dropped until the queue is drained. Userspace must resync its state upon receiving an overflow event.

## Subject ID Stability

Each `WatchEvent` carries a `subject_id` field that identifies the watched resource.  This is a **per-registration** ID assigned by the kernel at `sys_watch_fd` / `sys_watch_path` time.  IDs are assigned from a monotonically-increasing counter that is never reset or reused within a kernel session.

Key properties:
- `subject_id` is **not** a raw inode number.  Inode numbers are not stable across remounts and can be recycled; `subject_id` is immune to both hazards.
- Two `register_watch` calls on the same path/fd will produce **different** `subject_id` values.
- Clients may safely cache a `subject_id` returned at registration time and use it to correlate all subsequent events from that watch descriptor.
