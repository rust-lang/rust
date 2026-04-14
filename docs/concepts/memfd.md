# Memfd: the Bulk-Data Path

Thing-OS uses a **two-tier IPC model**: small control messages flow over
channels, and large or zero-copy data travels via memory-mapped things
(**memfd**).

## Doctrine: control over channels, bulk over memfd

| Concern | Mechanism | When to use |
|---|---|---|
| Commands, events, ACKs | `channel_send` / `channel_recv` | latency-sensitive; fits in a few hundred bytes |
| Pixel buffers, audio rings, large blobs | `memfd_create` + `vm_map` + `channel_send_msg` | throughput-sensitive; larger than a channel ring |

Never embed multi-kilobyte payloads in channel messages.  Instead, put the data
in a memfd and send the thing across the channel with
`channel_send_msg`.  The control message then carries only a small
[`abi::memfd::MemFdRef`] descriptor (16 bytes) that names the thing, and the
byte length of the valid window.

## Mapping mode semantics

`vm_map` supports both `MAP_SHARED` and `MAP_PRIVATE` semantics via
`VmMapFlags::SHARED` and `VmMapFlags::PRIVATE`:

* File-backed + `SHARED`: mappings reference the same physical pages (writes in
  one mapping become visible in peer mappings/processes).
* File-backed + `PRIVATE`: mapping is copy-on-map (new physical pages are
  allocated and initialized from file contents; writes are private).
* Anonymous mappings are currently private-only; requesting `SHARED` on
  anonymous backing returns `EOPNOTSUPP`.

For backward compatibility, callers that omit both flags default to
`SHARED` for file backing and `PRIVATE` for anonymous backing.

## Memfd lifetime and reference counting

The kernel maintains a reference count on each memfd region:

* A `memfd_create` call allocates contiguous physical frames and returns an
  open thing.
* Every `vm_map` call that backs itself on the thing increments the count.
* Calling `vfs_close(thing)` decrements the thing's reference.
* Calling `vm_unmap` decrements the mapping's reference.
* **Physical memory is freed only when the last thing and the last mapping are
  both gone.**

This means a receiver can safely hold onto its mapping even after the sender
has closed its own copy of the thing.

## Sharing a memfd across processes

```
Creator                          Receiver
───────                          ────────
memfd_create("frame", size)  →  thing

vm_map(thing, READ|WRITE|USER)  →  ptr
[fill pixel data at ptr]

channel_send_msg(chan, b"", &[thing])  ──────►  channel_recv_msg(chan) → new_thing
                                                 vm_map(new_thing, READ|USER) → ptr
                                                 [read pixel data at ptr]
                                                 vm_unmap(ptr, size)
                                                 vfs_close(new_thing)

vm_unmap(ptr, size)
vfs_close(thing)
           ╰── last reference dropped → physical pages freed
```

### Step-by-step (Rust / stem wrappers)

**Sender side**

```rust
use stem::syscall::{memfd_create, vm_map, channel_send_msg};
use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
use abi::memfd::MemFdRef;

let thing = memfd_create("frame", width * height * 4)?;

let req = VmMapReq {
    addr_hint: 0,
    len: width * height * 4,
    prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
  flags: VmMapFlags::SHARED,
    backing: VmBacking::File { fd: thing, offset: 0 },
};
let mapped = vm_map(&req)?;
let pixels: &mut [u32] = unsafe {
    core::slice::from_raw_parts_mut(mapped.addr as *mut u32, width * height)
};

// … fill pixels …

// Build the descriptor that will travel in the control message.
let desc = MemFdRef::new(thing, (width * height * 4) as u64);

// Encode the descriptor into the control message.
let mut ctrl = [0u8; 17];
ctrl[0] = MSG_PRESENT;
desc.encode_le(&mut ctrl[1..]).unwrap();

// Transfer the thing and bytes atomically.
channel_send_msg(channel, &ctrl, &[thing])?;
```

**Receiver side**

```rust
use stem::syscall::{channel_recv_msg, vm_map};
use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
use abi::memfd::MemFdRef;

// Receive the control bytes and the thing in one atomic call.
let mut ctrl = [0u8; 17];
let mut things = [0u32; 1];
let (_data_len, _thing_count) = channel_recv_msg(channel, &mut ctrl, &mut things)?;
let new_thing = things[0];

let desc = MemFdRef::decode_le(&ctrl[1..]).unwrap();

let req = VmMapReq {
    addr_hint: 0,
    len: desc.length as usize,
    prot: VmProt::READ | VmProt::USER,
  flags: VmMapFlags::SHARED,
    backing: VmBacking::File { fd: new_thing, offset: 0 },
};
let mapped = vm_map(&req)?;
let pixels: &[u32] = unsafe {
    core::slice::from_raw_parts(mapped.addr as *const u32, desc.length as usize / 4)
};

// … consume pixels …

vm_unmap(mapped.addr, desc.length as usize)?;
vfs_close(new_thing)?;
```

## Revocation

There is no forced-unmap primitive.  A sender signals "I am done with this
buffer" by closing its own thing and sending a companion control message (e.g.
`MSG_BUFFER_RELEASED`).  The receiver is responsible for unmapping promptly
when it receives that signal.  As long as the receiver holds its thing open,
the physical memory remains allocated — leaking is possible, so protocol
designers
must include an explicit release signal.

## Persistent shared rings

For streaming paths (audio, input, network) that need ultra-low latency without
repeated transfer handshakes, allocate the memfd once at setup and share it for
the lifetime of the session:

1. Server calls `memfd_create("ring", RING_BYTES)` and maps it read-write.
2. Server sends the thing to the client at connection time via
   `channel_send_msg`.
3. Client maps it and keeps the mapping open.
4. Both sides use an atomic sequence-number protocol (in the shared memory) to
   signal produce/consume positions — no thing transfers on the hot path.
5. When the session ends, both sides close their thing and unmap.

## Wire descriptor: `abi::memfd::MemFdRef`

All control-plane messages that accompany a memfd transfer **must** embed a
[`MemFdRef`] so the receiver can verify the expected length:

| Byte offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 4 | `fd` | Sender-local thing (receiver obtains its own via `channel_recv_msg`) |
| 4 | 4 | `_pad` | Reserved, must be zero |
| 8 | 8 | `length` | Byte length of the valid data window |

For pixel buffers, use [`abi::display::types::BufferHandle`] which extends this
with `width`, `height`, `stride`, `format`, and `modifier` fields.

## Existing in-tree users

| Crate | Pattern |
|---|---|
| `display_bootfb` | Bootstrap memfd passed as argv thing to deliver channel things |
| `display_virtio_gpu` | Frame-pool memfd created at startup; textures uploaded via per-frame memfds |
| `petals` | `Texture` type wraps a memfd + `vm_map` pointer for CPU-side pixel writes |
| `sprout` | Bootstrap memfd used to deliver driver channel things at spawn time |
| `blossom` SVG | `SvgSource::MemFd(thing)` transfers SVG bytes without copying into the IPC ring |
