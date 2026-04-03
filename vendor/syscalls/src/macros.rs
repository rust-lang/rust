/// Performs a syscall and returns a `Result<usize, Errno>`.
///
/// Accepts a syscall number and a variable number of arguments (0 to 6).
///
/// # Returns
///  - `Ok` on success, or
///  - `Err(errno)` if the syscall failed.
///
/// # Example
/// ```
/// use syscalls::{Sysno, syscall};
///
/// match unsafe { syscall!(Sysno::clone) } {
///     Ok(0) => {
///         // Child process
///     }
///     Ok(pid) => {
///         // Parent process
///     }
///     Err(err) => {
///         eprintln!("clone() failed: {}", err);
///     }
/// }
/// ```
#[macro_export]
macro_rules! syscall {
    ($nr:expr) => {
        $crate::syscall0($nr)
    };

    ($nr:expr, $a1:expr) => {
        $crate::syscall1($nr, $a1 as usize)
    };

    ($nr:expr, $a1:expr, $a2:expr) => {
        $crate::syscall2($nr, $a1 as usize, $a2 as usize)
    };

    ($nr:expr, $a1:expr, $a2:expr, $a3:expr) => {
        $crate::syscall3($nr, $a1 as usize, $a2 as usize, $a3 as usize)
    };

    ($nr:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr) => {
        $crate::syscall4(
            $nr,
            $a1 as usize,
            $a2 as usize,
            $a3 as usize,
            $a4 as usize,
        )
    };

    ($nr:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr) => {
        $crate::syscall5(
            $nr,
            $a1 as usize,
            $a2 as usize,
            $a3 as usize,
            $a4 as usize,
            $a5 as usize,
        )
    };

    ($nr:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr) => {
        $crate::syscall6(
            $nr,
            $a1 as usize,
            $a2 as usize,
            $a3 as usize,
            $a4 as usize,
            $a5 as usize,
            $a6 as usize,
        )
    };
}

/// Performs a raw syscall and returns a `usize`. Use [`syscall`] if you wish to
/// get a `Result` as a return value.
///
/// Accepts a syscall number and a variable number of arguments (0 to 6).
///
/// # Example
/// ```
/// use syscalls::{Sysno, raw_syscall};
///
/// // gettid is guaranteed to never fail, so we don't need a `Result` return
/// // value.
/// let tid = unsafe { raw_syscall!(Sysno::gettid) };
/// println!("My thread ID is {}", tid);
/// ```
#[macro_export]
macro_rules! raw_syscall {
    ($nr:expr) => {
        $crate::raw::syscall0($nr)
    };

    ($nr:expr, $a1:expr) => {
        $crate::raw::syscall1($nr, $a1 as usize)
    };

    ($nr:expr, $a1:expr, $a2:expr) => {
        $crate::raw::syscall2($nr, $a1 as usize, $a2 as usize)
    };

    ($nr:expr, $a1:expr, $a2:expr, $a3:expr) => {
        $crate::raw::syscall3($nr, $a1 as usize, $a2 as usize, $a3 as usize)
    };

    ($nr:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr) => {
        $crate::raw::syscall4(
            $nr,
            $a1 as usize,
            $a2 as usize,
            $a3 as usize,
            $a4 as usize,
        )
    };

    ($nr:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr) => {
        $crate::raw::syscall5(
            $nr,
            $a1 as usize,
            $a2 as usize,
            $a3 as usize,
            $a4 as usize,
            $a5 as usize,
        )
    };

    ($nr:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr) => {
        $crate::raw::syscall6(
            $nr,
            $a1 as usize,
            $a2 as usize,
            $a3 as usize,
            $a4 as usize,
            $a5 as usize,
            $a6 as usize,
        )
    };
}
