# thingos Service Namespace Convention

> *"Services do not register themselves with a central authority. They mount
> themselves into the namespace."* — Plan 9 inspiration

## Overview

In thingos, every service exposes its interface as a subtree in the VFS
namespace.  Clients discover services by opening well-known paths rather than
querying a registry.

A service provider is a normal userland process that:

1. Creates a port pair.
2. Calls `SYS_FS_MOUNT(write_handle, "/services/<name>")` to register as the
   filesystem provider for that subtree.
3. Enters a service loop, receiving [`abi::vfs_rpc`] messages and dispatching
   them to its internal state.

---

## Standard layout

```
/services/
  net/
    eth0/
      address      ← read/write: the current IPv4 address (text)
      status       ← read: "up" or "down"
      control      ← write: commands ("up", "down", "flush", …)
      events       ← read (blocking): newline-delimited event stream
  display/
    status         ← read: compositor state ("running", "off")
    socket         ← the Wayland/compositor socket path
  audio/
    pcm0/
      volume       ← read/write: "75" (percent, text)
      control      ← write: "play", "pause", "mute"
/run/
  wayland-0        ← Wayland socket (Bloom, when running)
/boot/
  iso/             ← ISO9660 filesystem from the boot CD-ROM (iso9660d)
    README
    BOOT/
      LIMINE.CFG
      …
```

---

## Standard files every service should expose

| Name      | Access    | Content                                              |
|-----------|-----------|------------------------------------------------------|
| `status`  | read      | One-line text: service state (e.g. `"up"`, `"idle"`) |
| `control` | write     | Short text commands (see below)                      |
| `events`  | read-block| Newline-delimited event records                      |

Services that do not implement a file should return `ENOENT` for its path;
callers must tolerate its absence.

---

## Control file convention

Services that accept commands expose a `control` file.  Clients write short
UTF-8 text commands terminated by `\n`:

```sh
echo "up"    > /services/net/eth0/control
echo "flush" > /services/net/eth0/control
```

Commands are single words (no arguments in v0).  A service may return
`EINVAL` for unrecognised commands.

### Well-known commands

| Command  | Meaning                                |
|----------|----------------------------------------|
| `up`     | Bring the resource up / enable it      |
| `down`   | Bring the resource down / disable it   |
| `flush`  | Flush cached state                     |
| `reset`  | Reset to default configuration         |
| `reload` | Re-read configuration from disk        |

---

## Event stream convention

Services that emit events expose an `events` file.  Readers block on it (via
`SYS_FS_READ` or `poll`) and receive newline-delimited records:

```
link-up eth0
link-down eth0
addr-changed eth0 192.168.1.42
```

Each line is a single event.  The format is:

```
<event-type> [<subject> [<value>]]
```

All fields are ASCII, space-separated.  Unknown events should be ignored by
clients (forward compatibility).

---

## Binary vs. text

- **Control paths** (`control`, `status`, `events`): always **UTF-8 text**.
- **Data streams** (e.g. raw audio PCM, network frames): **binary**, with the
  encoding documented per-service.
- **Configuration files**: UTF-8 text, `key=value` per line.

---

## VFS RPC protocol

Kernel→provider messages use the wire format defined in `abi/src/vfs_rpc.rs`.

Each request sent on the provider's port:
```
[resp_port: u32 LE] [op: u8] [_pad: u8 × 2] [payload…]
```

Each response sent by the provider to `resp_port`:
```
[status: u8]   0 = OK, else errno discriminant
[payload…]     present only on status == 0
```

### Operations

| Op       | Payload                                    | Response payload           |
|----------|--------------------------------------------|----------------------------|
| `Lookup` | `[path_len: u32][path: UTF-8]`             | `[handle: u64]`            |
| `Read`   | `[handle: u64][offset: u64][len: u32]`     | `[n: u32][data: n bytes]`  |
| `Write`  | `[handle: u64][offset: u64][len: u32][data]`| `[written: u32]`          |
| `Readdir`| `[handle: u64][offset: u64][len: u32]`     | `[n: u32][dirent data]`    |
| `Stat`   | `[handle: u64]`                            | `[mode: u32][size: u64][ino: u64]` |
| `Close`  | `[handle: u64]`                            | *(empty)*                  |
| `Poll`   | `[handle: u64][events: u32]`               | `[revents: u32]`           |

A `Lookup` for the empty string or `"/"` returns the root directory handle.

Handles are opaque `u64` values chosen by the provider.  The kernel passes
them back verbatim on `Read`, `Write`, `Readdir`, `Stat`, and `Close`.

---

## iso9660d: boot CD-ROM provider

`userspace/iso9660d` is the reference VFS provider implementation.  It:

1. Finds the first block device with a valid ISO9660 PVD.
2. Calls `vfs_mount(write_handle, "/boot/iso")`.
3. Serves `Lookup`, `Read`, `Readdir`, `Stat`, and `Close` using the
   `iso9660` library.  `Write` returns `EROFS`.

Files on the boot CD-ROM are accessible at paths like:

```
/boot/iso/README
/boot/iso/BOOT/LIMINE.CFG
```

---

## Adding a new service

1. Create `userspace/<name>d/`.
2. In `main`, create a port pair and call `vfs_mount(write_handle, "/services/<name>")`.
3. Implement a service loop that reads VFS RPC messages and responds.
4. Add an entry to this document.
