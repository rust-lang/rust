# Root Graph Batched Writes

## Overview
The Root Graph now supports high-throughput, atomic batched mutations via `SYS_ROOT_APPLY_BATCH`. This system allows userspace to submit a binary blob containing multiple graph operations (CreateNode, PutEdge, etc.) which are processed efficiently by the kernel.

## batch Format
The batch format is a linear binary stream of commands. Integers are Little Endian.

### Header
- `MAGIC` (4 bytes): `0x48544142` ("BATH")
- `VERSION` (4 bytes): `1`
- `OPS` (2 bytes): Number of operations in the batch.

### Operations
Each operation begins with a 1-byte `Tag`.

#### 1. OP_CREATE_NODE (Tag: 1)
- `Kind`: 16 bytes (SymbolId, or hex-interned string in v0).
- `OutRef`: 2 bytes (u16). The index in the local reference table to store this node's ID.

#### 2. OP_PUT_EDGE (Tag: 2)
- `Subject`: Reference (see below).
- `Relation`: 16 bytes (SymbolId).
- `Object`: Reference (see below).

### References
References identify a node.
- `REF_LOCAL` (Tag: 1): Followed by 2 bytes (u16) index into the batch's local node table.
- `REF_ABSOLUTE` (Tag: 2): Followed by 8 bytes (u64) GraphID.

## Syscalls

### SYS_ROOT_APPLY_BATCH (No. ?)
Applies a batch of operations.
Arguments:
- `buffer`: Pointer to the batch data.
- `len`: Length of the batch data.

Returns:
- `> 0`: The new Root Sequence ID (commit ID) upon success.
- `< 0`: Error code (e.g., `-EINVAL` for bad format).

## Watch Mechanism
Watches are sequence-based. Each commit produces a watch payload, not ApplyBatch bytes.
- `SYS_ROOT_WATCH_OPEN`: Returns a handle.
- `SYS_ROOT_WATCH_NEXT`: Reads the next watch payload for the next matching commit.
  - Returns watch payload bytes containing 0..N watch events (see `abi::watch`).
  - Returns `EOVERFLOW` if the queue overflowed (slow reader).
  - Returns `ENOBUFS` if the provided buffer is too small.

## Verification
Two userspace verification utilities are provided:

1. **root_batch_bench**: Benchmarks the throughput of batched writes.
   - Usage: `root_batch_bench`
   - Output: Throughput in Ops/Sec.

2. **root_watch_tester**: Verifies correctness of watch delivery, sequential ordering, and error handling (overflow, small buffer).
   - Usage: `root_watch_tester`
   - Output: PASS/FAIL results for various scenarios.
