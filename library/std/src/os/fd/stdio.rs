use super::BorrowedFd;

/// The file descriptor for the standard input stream of the current process.
///
/// See [`io::stdin()`][`crate::io::stdin`] for the higher level handle, which should be preferred
/// whenever possible. See [`STDERR`] for why the file descriptor might be required and caveats.
#[unstable(feature = "stdio_fd_consts", issue = "150836")]
pub const STDIN: BorrowedFd<'static> = unsafe { BorrowedFd::borrow_raw(0) };

/// The file descriptor for the standard output stream of the current process.
///
/// See [`io::stdout()`][`crate::io::stdout`] for the higher level handle, which should be preferred
/// whenever possible. See [`STDERR`] for why the file descriptor might be required and caveats. In
/// addition to the issues discussed there, note that [`Stdout`][`crate::io::Stdout`] is buffered by
/// default, and writing to the file descriptor will bypass this buffer.
#[unstable(feature = "stdio_fd_consts", issue = "150836")]
pub const STDOUT: BorrowedFd<'static> = unsafe { BorrowedFd::borrow_raw(1) };

/// The file descriptor for the standard error stream of the current process.
///
/// See [`io::stderr()`][`crate::io::stderr`] for the higher level handle, which should be preferred
/// whenever possible. However, there are situations where touching the `std::io` handles (or most
/// other parts of the standard library) risks deadlocks or other subtle bugs. For example:
///
/// - Global allocators must be careful to [avoid reentrancy][global-alloc-reentrancy], and the
///   `std::io` handles may allocate memory on (some) accesses.
/// - Signal handlers must be *async-signal-safe*, which rules out panicking, taking locks (may
///   deadlock if the signal handler interrupted a thread holding that lock), allocating memory, or
///   anything else that is not explicitly declared async-signal-safe.
/// - `CommandExt::pre_exec` callbacks can safely panic (with some limitations), but otherwise must
///   abide by similar limitations as signal handlers. In particular, at the time these callbacks
///   run, the stdio file descriptors have already been replaced, but the locks protecting the
///   `std::io` handles may be permanently locked if another thread held the lock at `fork()` time.
///
/// In these and similar cases, direct access to the file descriptor may be required. However, in
/// most cases, using the `std::io` handles and accessing the file descriptor via the `AsFd`
/// implementations is preferable, as it enables cooperation with the standard library's locking and
/// buffering.
///
/// # I/O safety
///
/// This is a `BorrowedFd<'static>` because the standard input/output/error streams are shared
/// resources that must remain available for the lifetime of the process. This is only true when
/// linking `std`, and may not always hold for [code running before `main()`][before-after-main] or
/// in `no_std` environments. It is [unsound][io-safety] to close these file descriptors. Safe
/// patterns for changing these file descriptors are available on Unix via the `StdioExt` extension
/// trait.
///
/// [before-after-main]: ../../../std/index.html#use-before-and-after-main
/// [io-safety]: ../../../std/io/index.html#io-safety
/// [global-alloc-reentrancy]: ../../../std/alloc/trait.GlobalAlloc.html#re-entrance
#[unstable(feature = "stdio_fd_consts", issue = "150836")]
pub const STDERR: BorrowedFd<'static> = unsafe { BorrowedFd::borrow_raw(2) };
