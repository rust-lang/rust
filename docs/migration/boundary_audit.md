# Kernel Boundary Type Audit

> **Status**: Phase 9 baseline audit — initial inventory artifact.
> **Companion**: `docs/migration/authority_inventory.md` (credential/permission axis)
> **Companion**: `docs/migration/bridge_architecture.md` (bridge layer design and conventions)
>
> This document inventories every **kernel-facing boundary surface** and records
> whether it currently uses kind-generated types as its canonical external shape.
>
> Sections follow the issue #41 deliverable format:
> boundary surface / current type / expected generated type / status / priority.

---

## Purpose

This document is the structured audit artifact for
_"Ensure Generated Types Are Used at All Kernel Boundaries"_.

It inventories the data shapes that cross the kernel–userspace boundary (or the
kernel's external observability boundary) and classifies each surface as:

| Status      | Meaning                                                                  |
|-------------|--------------------------------------------------------------------------|
| ✅ compliant  | generated type is already the canonical external shape                   |
| ⚠️ drifting  | hand-written boundary struct whose shape duplicates a generated type     |
| 📄 exception | cannot yet use generated type directly; reason documented                |

---

## 1. Bridge-Layer Boundaries (kernel → observation / IPC)

These surfaces are already fully wired through the canonical bridge pattern.
Each bridge module is the **single conversion point** from kernel-internal state
to the schema-generated public type.

| Boundary surface | Current type (kernel-internal) | Canonical generated type | Bridge module | Status |
|---|---|---|---|---|
| Task lifecycle state | `kernel::task::ThreadState` | `thingos::task::Task` | `kernel::task::bridge` | ✅ compliant |
| Job lifecycle | `kernel::sched::hooks::ProcessSnapshot` | `thingos::job::Job` | `kernel::job::bridge` | ✅ compliant |
| Job exit snapshot | `ProcessSnapshot` + `exit_code` | `thingos::job::JobExit` | `kernel::job::bridge` | ✅ compliant |
| Job wait result | `poll_task_exit` → `Option<i32>` | `thingos::job::JobWaitResult` | `kernel::job::bridge` | ✅ compliant |
| Coordination domain | `Process::unix_compat.pgid` + TTY state | `thingos::group::Group` | `kernel::group::bridge` | ✅ compliant |
| Permission context | `ProcessSnapshot::name` | `thingos::authority::Authority` | `kernel::authority::bridge` | ✅ compliant |
| World context | `Process::cwd` + `Process::namespace` | `thingos::place::Place` | `kernel::place::bridge` | ✅ compliant |
| IPC message envelope | raw `(kind_id: [u8;16], payload: Vec<u8>)` | `thingos::message::Message` | `kernel::message::bridge` | ✅ compliant |

### Guardrail

`kernel/src/boundary_contract.rs` contains two compile-time + runtime tests
that enforce the above:

1. **`bridge_signatures_return_canonical_types`** — type-checks all bridge
   function signatures via let bindings.  Changing a bridge return type away
   from the canonical type becomes a compile error.

2. **`kind_ids_match_kindc_generated_constants`** — parses the kindc fixture
   file (`tools/kindc/fixtures/generated/mod.rs`) at test time and asserts that
   every `KIND_ID_THINGOS_*` constant in the `thingos` crate matches the value
   produced by the schema compiler.  Schema drift between kindc output and the
   hand-maintained constants becomes a test failure.

---

## 2. KindId Constant Alignment

Every schema kind that has a boundary-relevant Rust type also exposes a
`KIND_ID_THINGOS_*` constant in its owning module.  These constants are the
stable schema-version identifiers used for message dispatch.

| Constant | Module | Enforced by boundary_contract? |
|---|---|---|
| `KIND_ID_THINGOS_MESSAGE` | `thingos::message::KindId::THINGOS_MESSAGE` | ✅ yes |
| `KIND_ID_THINGOS_JOB_EXIT` | `thingos::message::KindId::THINGOS_JOB_EXIT` | ✅ yes |
| `KIND_ID_THINGOS_AUTHORITY` | `thingos::authority` | ✅ yes |
| `KIND_ID_THINGOS_TASK` | `thingos::task` | ✅ yes |
| `KIND_ID_THINGOS_TASK_STATE` | `thingos::task` | ✅ yes |
| `KIND_ID_THINGOS_JOB` | `thingos::job` | ✅ yes |
| `KIND_ID_THINGOS_JOB_STATE` | `thingos::job` | ✅ yes |
| `KIND_ID_THINGOS_JOB_EXIT` | `thingos::job` | ✅ yes |
| `KIND_ID_THINGOS_JOB_WAIT_RESULT` | `thingos::job` | ✅ yes |
| `KIND_ID_THINGOS_GROUP` | `thingos::group` | ✅ yes |
| `KIND_ID_THINGOS_PLACE` | `thingos::place` | ✅ yes |

---

## 3. VFS RPC Boundary

The kernel–provider VFS RPC protocol uses hand-written `abi::vfs_rpc` types:
`VfsRpcReqHeader`, `DirentWire`, `VfsRpcOp`, etc.

| Boundary surface | Current type | Generated equivalent | Status |
|---|---|---|---|
| VFS RPC request header | `abi::vfs_rpc::VfsRpcReqHeader` | none yet | 📄 exception |
| VFS RPC directory entry | `abi::vfs_rpc::DirentWire` | none yet | 📄 exception |
| VFS RPC operation codes | `abi::vfs_rpc::VfsRpcOp` | none yet | 📄 exception |
| Provider request decode | `libs/ipc_helpers::provider::ProviderRequest` | none yet | 📄 exception |

**Exception reason**: The VFS RPC wire protocol is a low-level binary layout
(`#[repr(C, packed)]`) with explicit padding fields.  No kindc schema kind has
been defined for it yet.  The existing `abi` types are the canonical source of
truth for this layer.  A future task should define VFS RPC kinds and regenerate
these types, at which point this exception can be resolved.

---

## 4. Syscall Raw Argument Types

Syscall request/response structs passed directly across the user/kernel pointer
boundary are defined in `abi::types` and `abi::syscall`.

| Boundary surface | Current type | Generated equivalent | Status |
|---|---|---|---|
| Thread spawn request | `abi::types::SpawnThreadReq` | none yet | 📄 exception |
| Process spawn args | `abi::types::SpawnProcessExArgs` | none yet | 📄 exception |
| VM map/protect args | `abi::vm::VmMapArgs` etc. | none yet | 📄 exception |
| Stat structure | `abi::syscall::Stat` | none yet | 📄 exception |
| Poll fd set | `abi::syscall::PollFd` | none yet | 📄 exception |

**Exception reason**: These types are C-ABI-compatible structures passed as raw
pointers in syscall arguments.  They require precise `#[repr(C)]` layout
guarantees and explicit size/alignment constraints that kindc-generated types
do not yet guarantee.  Until the schema compiler supports generating ABI-stable
C-layout structs with explicit field offsets and padding, these remain as
hand-written `abi` types.

---

## 5. Typed Message Delivery Boundary

The `SYS_MSG_SEND` and `SYS_MSG_BROADCAST` syscall handlers decode raw user
pointers into a canonical `thingos::message::Message`.

| Boundary surface | Current decode | Canonical target | Status |
|---|---|---|---|
| `sys_msg_send` | `copyin_message()` → `Message::new(kind, payload)` | `thingos::message::Message` | ✅ compliant |
| `sys_msg_broadcast` | `copyin_message()` → `Message::new(kind, payload)` | `thingos::message::Message` | ✅ compliant |
| Inbox envelope | `MessageEnvelope { message: Message, sender }` | wraps `thingos::message::Message` | ✅ compliant |
| Job exit notification | `JobExit::encode_as_notification()` → `Message` | `thingos::message::Message` | ✅ compliant |

---

## 6. Structural Divergence Notes

Several `thingos` crate types intentionally diverge from the kindc-generated
versions in `tools/kindc/fixtures/generated/mod.rs`.  This is expected during
the transitional migration period.

| Type | thingos crate shape | Generated shape | Reason for divergence |
|---|---|---|---|
| `Task::job` | `Option<u32>` (PID as stand-in) | `Option<ThingId>` | First-class `ThingId` for jobs not yet available |
| `Place` fields | `cwd`, `namespace`, `root` | `name`, `kind: PlaceKind` | thingos schema evolved beyond generated snapshot |
| `Group::kind` | `GroupKind` (Foreground/Coordination) | `members: Vec<ThingId>, name` | thingos schema evolved beyond generated snapshot |

These divergences are tracked here so they remain visible rather than mythical.
The **KindId constants** remain aligned between the two, anchoring the semantic
versioning even while structural details evolve.  A future `just kindc-gen` pass
should regenerate the fixture file from the current schemas and re-sync the
generated types.

---

## 7. Exception Summary

| Boundary area | Exception reason | Follow-up task |
|---|---|---|
| VFS RPC protocol (`abi::vfs_rpc`) | No kindc schema defined; C-packed layout | Define VFS RPC kinds in kindc, regenerate |
| Syscall raw ABI structs (`abi::types`) | C-ABI layout guarantees; no kindc C-layout support | Add `#[repr(C)]` generation to kindc or keep as `abi` crate |
| `Task::job` type (`Option<u32>`) | No first-class `ThingId` for jobs yet | Replace when Job becomes a first-class kernel object |
| `Place` / `Group` structural shape | Schema evolved past kindc snapshot | Re-run `just kindc-gen` to sync fixture |

---

## 8. Regression Prevention

The primary guardrail is `kernel/src/boundary_contract.rs`, which:

1. **Compile-time checks**: bridge function signatures are type-checked by
   assigning them to explicitly-typed `fn` pointers.  Changing a signature away
   from the canonical type produces a compiler error.

2. **Runtime checks** (`cargo test -p kernel`): all `KIND_ID_THINGOS_*`
   constants are compared against the kindc fixture file.  Any hand-maintained
   constant that drifts from the generated value causes a test failure.

Additional conventions:

- New boundary-facing functions **must** be added to `boundary_contract.rs`
  before merging.
- New schema kinds with boundary relevance **must** expose a `KIND_ID_THINGOS_*`
  constant and be included in the `kind_ids_match_kindc_generated_constants` test.
- Hand-written structs at boundary entry points that duplicate a generated type
  are **not permitted**.  Use the canonical type or document the exception here.
