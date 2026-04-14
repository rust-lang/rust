# Supervisor Registration and Service-Publication Protocol

This document describes the typed channel-based protocol used by drivers and
services to register themselves with the **sprout** supervisor and to publish
their VFS provider into the device namespace.

---

## Overview

When a driver is launched by sprout (or by `devd`), it receives a pair of
private channel things as part of its bootstrap data:

| Thing           | Direction         | Purpose                                 |
|-----------------|-------------------|-----------------------------------------|
| `drv_req_read`  | Supervisor → Driver | Commands and replies from the supervisor |
| `drv_resp_write`| Driver → Supervisor | Registration messages and status updates |

All messages follow the framing defined in `abi::display_driver_protocol`
(a simple 4-byte little-endian length prefix + 2-byte message type).

---

## Message Catalogue

| Constant              | Value  | Direction             | Meaning                                       |
|-----------------------|--------|-----------------------|-----------------------------------------------|
| `MSG_BIND_READY`      | 0x8001 | Driver → Supervisor   | Driver ready; VFS provider thing attached     |
| `MSG_BIND_ASSIGNED`   | 0x8002 | Supervisor → Driver   | Registration accepted; path assigned          |
| `MSG_BIND_FAILED`     | 0x8003 | Supervisor → Driver   | Registration rejected; error details included |
| `MSG_SERVICE_READY`   | 0x8004 | Driver → Supervisor   | Service fully operational                     |
| `MSG_SERVICE_EXITING` | 0x8005 | Driver → Supervisor   | Service shutting down cleanly                 |

All types and encode/decode helpers are in `abi::supervisor_protocol`.

---

## Handshake Sequence

```
Driver                                   Supervisor (sprout)
  │                                             │
  │── channel_send_msg(vfs_write+MSG_BIND_READY) ─▶│  (VFS provider thing + BindReadyPayload)
  │                                             │
  │                                ┌────────────┤
  │                                │ Verify:    │
  │                                │ • thing    │
  │                                │ • class_mask│
  │                                │ • vfs_mount │
  │                                └────────────┤
  │                                             │
  │◀── MSG_BIND_ASSIGNED (success) ─────────────│  (BindAssignedPayload)
  │    OR                                       │
  │◀── MSG_BIND_FAILED   (failure) ─────────────│  (BindFailedPayload)
  │                                             │
  │── (initialize service internals) ───────────│
  │                                             │
  │── MSG_SERVICE_READY ───────────────────────▶│  (ServiceReadyPayload)
  │                                             │
  │  … normal operation …                       │
  │                                             │
  │── MSG_SERVICE_EXITING ─────────────────────▶│  (ServiceExitingPayload)
  │                                             │
```

### Step 1: Send `MSG_BIND_READY`

The driver creates a VFS provider channel pair and sends:

1. The **write** thing of the VFS provider channel via `channel_send_msg` on
   `drv_resp_write`.
2. A `MSG_BIND_READY` message with a `BindReadyPayload` on `drv_resp_write`.

```rust
use abi::supervisor_protocol::{self, classes};

let (vfs_write, vfs_read) = channel_create(VFS_RPC_MAX_REQ * 8)?;

let ready = supervisor_protocol::BindReadyPayload {
    bind_instance_id,              // token issued by supervisor at launch
    class_mask: classes::DISPLAY_CARD | classes::FRAMEBUFFER,
    _reserved: 0,
};
let mut payload_bytes = [0u8; supervisor_protocol::BIND_READY_PAYLOAD_SIZE];
supervisor_protocol::encode_bind_ready_le(&ready, &mut payload_bytes);

let mut buf = [0u8; 256];
let msg_len = display_driver_protocol::encode_message(
    &mut buf, supervisor_protocol::MSG_BIND_READY, &payload_bytes,
).unwrap();

channel_send_msg(drv_resp_write, &buf[..msg_len], &[vfs_write]).unwrap(); // thing + bytes together
```

### Step 2: Wait for `MSG_BIND_ASSIGNED` or `MSG_BIND_FAILED`

The driver polls `drv_req_read`:

```rust
loop {
    if let Ok(n) = channel_try_recv(drv_req_read, &mut buf) {
        if let Some((hdr, payload)) = display_driver_protocol::parse_message(&buf[..n]) {
            if hdr.msg_type == supervisor_protocol::MSG_BIND_ASSIGNED {
                let assigned = supervisor_protocol::decode_bind_assigned_le(payload)?;
                // Use assigned.primary_path, assigned.unit_number …
                break;
            } else if hdr.msg_type == supervisor_protocol::MSG_BIND_FAILED {
                let failed = supervisor_protocol::decode_bind_failed_le(payload)?;
                // Log error and halt; supervisor rejected the registration.
                loop { yield_now(); }
            }
        }
    }
    yield_now();
}
```

### Step 3: Send `MSG_SERVICE_READY`

After completing internal initialisation (e.g. scanning hardware, allocating
buffers, mounting sub-trees), the driver notifies the supervisor:

```rust
let svc_ready = supervisor_protocol::ServiceReadyPayload {
    bind_instance_id,
    _reserved: 0,
};
// encode and send on drv_resp_write …
```

### Step 4: Send `MSG_SERVICE_EXITING` (on clean shutdown)

Before the process exits, it should inform the supervisor so it can
distinguish a clean shutdown from an unexpected crash:

```rust
let exiting = supervisor_protocol::ServiceExitingPayload {
    bind_instance_id,
    exit_code: 0,
    _reserved: 0,
};
// encode and send on drv_resp_write …
```

---

## Supervisor Verification Logic

When sprout receives `MSG_BIND_READY` it performs the following checks in
order, sending `MSG_BIND_FAILED` on the first failure:

| Step | Check                                  | Error code                  |
|------|----------------------------------------|-----------------------------|
| 1    | `decode_bind_ready_le` succeeds        | `ERR_INVALID_MESSAGE` (1)   |
| 2    | `channel_recv_msg` returns a thing     | `ERR_NO_PROVIDER_HANDLE` (2)|
| 3    | `class_mask != 0` and is recognised    | `ERR_UNKNOWN_CLASS` (3)     |
| 4    | `vfs_mount(thing, path)` succeeds      | `ERR_MOUNT_FAILED` (4)      |

On success, sprout:

1. Allocates a canonical path under `/dev/<class>/` using a per-class
   monotonic counter tracked in its internal `DeviceLedger`.
2. Calls `vfs_mount(provider_thing, path)` to publish the driver's VFS
   subtree.
3. Replies with `MSG_BIND_ASSIGNED` carrying the assigned path and unit
   number.

---

## Payload Sizes

| Payload                | Rust type              | Wire size |
|------------------------|------------------------|-----------|
| `BindReadyPayload`     | `BindReadyPayload`     | 16 bytes  |
| `BindAssignedPayload`  | `BindAssignedPayload`  | 80 bytes  |
| `BindFailedPayload`    | `BindFailedPayload`    | 80 bytes  |
| `ServiceReadyPayload`  | `ServiceReadyPayload`  | 16 bytes  |
| `ServiceExitingPayload`| `ServiceExitingPayload`| 16 bytes  |

All multi-byte fields are little-endian.

---

## Class Bits

```rust
pub mod classes {
    pub const DISPLAY_CARD:      u32 = 1 << 0;   // /dev/display/card{N}
    pub const FRAMEBUFFER:       u32 = 1 << 1;   // (combined with DISPLAY_CARD)
    pub const INPUT_EVENT:       u32 = 1 << 2;   // /dev/input/event{N}
    pub const BLOCK_DEVICE:      u32 = 1 << 3;   // /dev/block/sd{N}
    pub const NETWORK_INTERFACE: u32 = 1 << 4;   // /dev/net/virtio{N}
    pub const SOUND_CARD:        u32 = 1 << 5;   // /dev/sound/card{N}
}
```

Drivers with no recognised class bit receive `MSG_BIND_FAILED` with
`ERR_UNKNOWN_CLASS`.

---

## Error Codes

| Constant                  | Value | Meaning                                      |
|---------------------------|-------|----------------------------------------------|
| `ERR_INVALID_MESSAGE`     | 1     | `BIND_READY` payload could not be decoded    |
| `ERR_NO_PROVIDER_HANDLE`  | 2     | No VFS provider thing attached               |
| `ERR_UNKNOWN_CLASS`       | 3     | `class_mask` is zero or unrecognised         |
| `ERR_MOUNT_FAILED`        | 4     | Kernel `vfs_mount` call failed               |

---

## See Also

- `abi/src/supervisor_protocol.rs` — protocol types and encode/decode helpers
- `userspace/sprout/src/supervisor.rs` — supervisor implementation
- `drivers/display_bootfb/src/main.rs` — reference driver using this protocol
- `drivers/display_virtio_gpu/src/main.rs` — reference driver (GPU path)
- `drivers/virtio_netd/src/main.rs` — reference driver (network path)
- `docs/services-layout.md` — VFS namespace layout
