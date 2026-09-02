Volatile operations are intended to act on I/O memory. As such, they are considered externally
observable events (just like syscalls, but less opaque), and are guaranteed to not be elided or
reordered by the compiler across other externally observable events. With this in mind, there
are two cases of usage that need to be distinguished:

- When a volatile operation is used for memory inside an [allocation], it behaves exactly
  like [`load`][Self::load], except for the additional guarantee that it won't be elided or
  reordered across other externally observable events (see above). This implies that the
  operation will actually access memory and not e.g. be lowered to reusing data from a
  previous load. Other than that, all the usual rules for memory accesses apply (including
  provenance).

- Volatile operations, however, may also be used to access memory that is _outside_ of any Rust
  allocation. In this use-case, the pointer does *not* have to be [valid] for reads. This is
  typically used for CPU and peripheral registers that must be accessed via an I/O memory mapping,
  most commonly at fixed addresses reserved by the hardware. These often have special semantics
  associated to their manipulation, and cannot be used as general purpose memory. Here, any address
  value is possible, including 0 and [`usize::MAX`], so long as the semantics of such a read are
  well-defined by the target hardware. The provenance of the pointer is irrelevant, and it can be
  created with [`without_provenance`][crate::ptr::without_provenance]. The access is allowed to
  trap, which must immediately abort the process. It can also cause other side-effects, but those
  must not affect Rust-allocated memory in any way.

In both cases, the access is also considered atomic with the given `order`. This allows
synchronization with other threads or devices that share memory with this program.

When invoked during const evaluation, this behaves like a regular atomic load. In
particular, such reads must always follow the first of the two cases above.

[allocation]: crate::ptr#allocated-object
