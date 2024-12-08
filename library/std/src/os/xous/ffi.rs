#![allow(dead_code)]
#![allow(unused_variables)]
#![stable(feature = "rust1", since = "1.0.0")]

#[path = "../unix/ffi/os_str.rs"]
mod os_str;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::os_str::{OsStrExt, OsStringExt};

mod definitions;
#[stable(feature = "rust1", since = "1.0.0")]
pub use definitions::*;

fn lend_mut_impl(
    connection: Connection,
    opcode: usize,
    data: &mut [u8],
    arg1: usize,
    arg2: usize,
    blocking: bool,
) -> Result<(usize, usize), Error> {
    let mut a0 = if blocking { Syscall::SendMessage } else { Syscall::TrySendMessage } as usize;
    let mut a1: usize = connection.try_into().unwrap();
    let mut a2 = InvokeType::LendMut as usize;
    let a3 = opcode;
    let a4 = data.as_mut_ptr() as usize;
    let a5 = data.len();
    let a6 = arg1;
    let a7 = arg2;

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0,
            inlateout("a1") a1,
            inlateout("a2") a2,
            inlateout("a3") a3 => _,
            inlateout("a4") a4 => _,
            inlateout("a5") a5 => _,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };

    let result = a0;

    if result == SyscallResult::MemoryReturned as usize {
        Ok((a1, a2))
    } else if result == SyscallResult::Error as usize {
        Err(a1.into())
    } else {
        Err(Error::InternalError)
    }
}

pub(crate) fn lend_mut(
    connection: Connection,
    opcode: usize,
    data: &mut [u8],
    arg1: usize,
    arg2: usize,
) -> Result<(usize, usize), Error> {
    lend_mut_impl(connection, opcode, data, arg1, arg2, true)
}

pub(crate) fn try_lend_mut(
    connection: Connection,
    opcode: usize,
    data: &mut [u8],
    arg1: usize,
    arg2: usize,
) -> Result<(usize, usize), Error> {
    lend_mut_impl(connection, opcode, data, arg1, arg2, false)
}

fn lend_impl(
    connection: Connection,
    opcode: usize,
    data: &[u8],
    arg1: usize,
    arg2: usize,
    blocking: bool,
) -> Result<(usize, usize), Error> {
    let mut a0 = if blocking { Syscall::SendMessage } else { Syscall::TrySendMessage } as usize;
    let a1: usize = connection.try_into().unwrap();
    let a2 = InvokeType::Lend as usize;
    let a3 = opcode;
    let a4 = data.as_ptr() as usize;
    let a5 = data.len();
    let a6 = arg1;
    let a7 = arg2;
    let mut ret1;
    let mut ret2;

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0,
            inlateout("a1") a1 => ret1,
            inlateout("a2") a2 => ret2,
            inlateout("a3") a3 => _,
            inlateout("a4") a4 => _,
            inlateout("a5") a5 => _,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };

    let result = a0;

    if result == SyscallResult::MemoryReturned as usize {
        Ok((ret1, ret2))
    } else if result == SyscallResult::Error as usize {
        Err(ret1.into())
    } else {
        Err(Error::InternalError)
    }
}

pub(crate) fn lend(
    connection: Connection,
    opcode: usize,
    data: &[u8],
    arg1: usize,
    arg2: usize,
) -> Result<(usize, usize), Error> {
    lend_impl(connection, opcode, data, arg1, arg2, true)
}

pub(crate) fn try_lend(
    connection: Connection,
    opcode: usize,
    data: &[u8],
    arg1: usize,
    arg2: usize,
) -> Result<(usize, usize), Error> {
    lend_impl(connection, opcode, data, arg1, arg2, false)
}

fn scalar_impl(connection: Connection, args: [usize; 5], blocking: bool) -> Result<(), Error> {
    let mut a0 = if blocking { Syscall::SendMessage } else { Syscall::TrySendMessage } as usize;
    let mut a1: usize = connection.try_into().unwrap();
    let a2 = InvokeType::Scalar as usize;
    let a3 = args[0];
    let a4 = args[1];
    let a5 = args[2];
    let a6 = args[3];
    let a7 = args[4];

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0,
            inlateout("a1") a1,
            inlateout("a2") a2 => _,
            inlateout("a3") a3 => _,
            inlateout("a4") a4 => _,
            inlateout("a5") a5 => _,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };

    let result = a0;

    if result == SyscallResult::Ok as usize {
        Ok(())
    } else if result == SyscallResult::Error as usize {
        Err(a1.into())
    } else {
        Err(Error::InternalError)
    }
}

pub(crate) fn scalar(connection: Connection, args: [usize; 5]) -> Result<(), Error> {
    scalar_impl(connection, args, true)
}

pub(crate) fn try_scalar(connection: Connection, args: [usize; 5]) -> Result<(), Error> {
    scalar_impl(connection, args, false)
}

fn blocking_scalar_impl(
    connection: Connection,
    args: [usize; 5],
    blocking: bool,
) -> Result<[usize; 5], Error> {
    let mut a0 = if blocking { Syscall::SendMessage } else { Syscall::TrySendMessage } as usize;
    let mut a1: usize = connection.try_into().unwrap();
    let mut a2 = InvokeType::BlockingScalar as usize;
    let mut a3 = args[0];
    let mut a4 = args[1];
    let mut a5 = args[2];
    let a6 = args[3];
    let a7 = args[4];

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0,
            inlateout("a1") a1,
            inlateout("a2") a2,
            inlateout("a3") a3,
            inlateout("a4") a4,
            inlateout("a5") a5,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };

    let result = a0;

    if result == SyscallResult::Scalar1 as usize {
        Ok([a1, 0, 0, 0, 0])
    } else if result == SyscallResult::Scalar2 as usize {
        Ok([a1, a2, 0, 0, 0])
    } else if result == SyscallResult::Scalar5 as usize {
        Ok([a1, a2, a3, a4, a5])
    } else if result == SyscallResult::Error as usize {
        Err(a1.into())
    } else {
        Err(Error::InternalError)
    }
}

pub(crate) fn blocking_scalar(
    connection: Connection,
    args: [usize; 5],
) -> Result<[usize; 5], Error> {
    blocking_scalar_impl(connection, args, true)
}

pub(crate) fn try_blocking_scalar(
    connection: Connection,
    args: [usize; 5],
) -> Result<[usize; 5], Error> {
    blocking_scalar_impl(connection, args, false)
}

fn connect_impl(address: ServerAddress, blocking: bool) -> Result<Connection, Error> {
    let a0 = if blocking { Syscall::Connect } else { Syscall::TryConnect } as usize;
    let address: [u32; 4] = address.into();
    let a1: usize = address[0].try_into().unwrap();
    let a2: usize = address[1].try_into().unwrap();
    let a3: usize = address[2].try_into().unwrap();
    let a4: usize = address[3].try_into().unwrap();
    let a5 = 0;
    let a6 = 0;
    let a7 = 0;

    let mut result: usize;
    let mut value: usize;

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0 => result,
            inlateout("a1") a1 => value,
            inlateout("a2") a2 => _,
            inlateout("a3") a3 => _,
            inlateout("a4") a4 => _,
            inlateout("a5") a5 => _,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };
    if result == SyscallResult::ConnectionId as usize {
        Ok(value.try_into().unwrap())
    } else if result == SyscallResult::Error as usize {
        Err(value.into())
    } else {
        Err(Error::InternalError)
    }
}

/// Connects to a Xous server represented by the specified `address`.
///
/// The current thread will block until the server is available. Returns
/// an error if the server cannot accept any more connections.
pub(crate) fn connect(address: ServerAddress) -> Result<Connection, Error> {
    connect_impl(address, true)
}

/// Attempts to connect to a Xous server represented by the specified `address`.
///
/// If the server does not exist then None is returned.
pub(crate) fn try_connect(address: ServerAddress) -> Result<Option<Connection>, Error> {
    match connect_impl(address, false) {
        Ok(conn) => Ok(Some(conn)),
        Err(Error::ServerNotFound) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Terminates the current process and returns the specified code to the parent process.
pub(crate) fn exit(return_code: u32) -> ! {
    let a0 = Syscall::TerminateProcess as usize;
    let a1 = return_code as usize;
    let a2 = 0;
    let a3 = 0;
    let a4 = 0;
    let a5 = 0;
    let a6 = 0;
    let a7 = 0;

    unsafe {
        core::arch::asm!(
            "ecall",
            in("a0") a0,
            in("a1") a1,
            in("a2") a2,
            in("a3") a3,
            in("a4") a4,
            in("a5") a5,
            in("a6") a6,
            in("a7") a7,
        )
    };
    unreachable!();
}

/// Suspends the current thread and allow another thread to run. This thread may
/// continue executing again immediately if there are no other threads available
/// to run on the system.
pub(crate) fn do_yield() {
    let a0 = Syscall::Yield as usize;
    let a1 = 0;
    let a2 = 0;
    let a3 = 0;
    let a4 = 0;
    let a5 = 0;
    let a6 = 0;
    let a7 = 0;

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0 => _,
            inlateout("a1") a1 => _,
            inlateout("a2") a2 => _,
            inlateout("a3") a3 => _,
            inlateout("a4") a4 => _,
            inlateout("a5") a5 => _,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };
}

/// Allocates memory from the system.
///
/// An optional physical and/or virtual address may be specified in order to
/// ensure memory is allocated at specific offsets, otherwise the kernel will
/// select an address.
///
/// # Safety
///
/// This function is safe unless a virtual address is specified. In that case,
/// the kernel will return an alias to the existing range. This violates Rust's
/// pointer uniqueness guarantee.
pub(crate) unsafe fn map_memory<T>(
    phys: Option<core::ptr::NonNull<T>>,
    virt: Option<core::ptr::NonNull<T>>,
    count: usize,
    flags: MemoryFlags,
) -> Result<&'static mut [T], Error> {
    let mut a0 = Syscall::MapMemory as usize;
    let mut a1 = phys.map(|p| p.as_ptr() as usize).unwrap_or_default();
    let mut a2 = virt.map(|p| p.as_ptr() as usize).unwrap_or_default();
    let a3 = count * core::mem::size_of::<T>();
    let a4 = flags.bits();
    let a5 = 0;
    let a6 = 0;
    let a7 = 0;

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0,
            inlateout("a1") a1,
            inlateout("a2") a2,
            inlateout("a3") a3 => _,
            inlateout("a4") a4 => _,
            inlateout("a5") a5 => _,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };

    let result = a0;

    if result == SyscallResult::MemoryRange as usize {
        let start = core::ptr::with_exposed_provenance_mut::<T>(a1);
        let len = a2 / core::mem::size_of::<T>();
        let end = unsafe { start.add(len) };
        Ok(unsafe { core::slice::from_raw_parts_mut(start, len) })
    } else if result == SyscallResult::Error as usize {
        Err(a1.into())
    } else {
        Err(Error::InternalError)
    }
}

/// Destroys the given memory, returning it to the compiler.
///
/// Safety: The memory pointed to by `range` should not be used after this
/// function returns, even if this function returns Err().
pub(crate) unsafe fn unmap_memory<T>(range: *mut [T]) -> Result<(), Error> {
    let mut a0 = Syscall::UnmapMemory as usize;
    let mut a1 = range.as_mut_ptr() as usize;
    let a2 = range.len() * core::mem::size_of::<T>();
    let a3 = 0;
    let a4 = 0;
    let a5 = 0;
    let a6 = 0;
    let a7 = 0;

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0,
            inlateout("a1") a1,
            inlateout("a2") a2 => _,
            inlateout("a3") a3 => _,
            inlateout("a4") a4 => _,
            inlateout("a5") a5 => _,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };

    let result = a0;

    if result == SyscallResult::Ok as usize {
        Ok(())
    } else if result == SyscallResult::Error as usize {
        Err(a1.into())
    } else {
        Err(Error::InternalError)
    }
}

/// Adjusts the memory flags for the given range.
///
/// This can be used to remove flags from a given region in order to harden
/// memory access. Note that flags may only be removed and may never be added.
///
/// Safety: The memory pointed to by `range` may become inaccessible or have its
/// mutability removed. It is up to the caller to ensure that the flags specified
/// by `new_flags` are upheld, otherwise the program will crash.
pub(crate) unsafe fn update_memory_flags<T>(
    range: *mut [T],
    new_flags: MemoryFlags,
) -> Result<(), Error> {
    let mut a0 = Syscall::UpdateMemoryFlags as usize;
    let mut a1 = range.as_mut_ptr() as usize;
    let a2 = range.len() * core::mem::size_of::<T>();
    let a3 = new_flags.bits();
    let a4 = 0; // Process ID is currently None
    let a5 = 0;
    let a6 = 0;
    let a7 = 0;

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0,
            inlateout("a1") a1,
            inlateout("a2") a2 => _,
            inlateout("a3") a3 => _,
            inlateout("a4") a4 => _,
            inlateout("a5") a5 => _,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };

    let result = a0;

    if result == SyscallResult::Ok as usize {
        Ok(())
    } else if result == SyscallResult::Error as usize {
        Err(a1.into())
    } else {
        Err(Error::InternalError)
    }
}

/// Creates a thread with a given stack and up to four arguments.
pub(crate) fn create_thread(
    start: *mut usize,
    stack: *mut [u8],
    arg0: usize,
    arg1: usize,
    arg2: usize,
    arg3: usize,
) -> Result<ThreadId, Error> {
    let mut a0 = Syscall::CreateThread as usize;
    let mut a1 = start as usize;
    let a2 = stack.as_mut_ptr() as usize;
    let a3 = stack.len();
    let a4 = arg0;
    let a5 = arg1;
    let a6 = arg2;
    let a7 = arg3;

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0,
            inlateout("a1") a1,
            inlateout("a2") a2 => _,
            inlateout("a3") a3 => _,
            inlateout("a4") a4 => _,
            inlateout("a5") a5 => _,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };

    let result = a0;

    if result == SyscallResult::ThreadId as usize {
        Ok(a1.into())
    } else if result == SyscallResult::Error as usize {
        Err(a1.into())
    } else {
        Err(Error::InternalError)
    }
}

/// Waits for the given thread to terminate and returns the exit code from that thread.
pub(crate) fn join_thread(thread_id: ThreadId) -> Result<usize, Error> {
    let mut a0 = Syscall::JoinThread as usize;
    let mut a1 = thread_id.into();
    let a2 = 0;
    let a3 = 0;
    let a4 = 0;
    let a5 = 0;
    let a6 = 0;
    let a7 = 0;

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0,
            inlateout("a1") a1,
            inlateout("a2") a2 => _,
            inlateout("a3") a3 => _,
            inlateout("a4") a4 => _,
            inlateout("a5") a5 => _,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };

    let result = a0;

    if result == SyscallResult::Scalar1 as usize {
        Ok(a1)
    } else if result == SyscallResult::Scalar2 as usize {
        Ok(a1)
    } else if result == SyscallResult::Scalar5 as usize {
        Ok(a1)
    } else if result == SyscallResult::Error as usize {
        Err(a1.into())
    } else {
        Err(Error::InternalError)
    }
}

/// Gets the current thread's ID.
pub(crate) fn thread_id() -> Result<ThreadId, Error> {
    let mut a0 = Syscall::GetThreadId as usize;
    let mut a1 = 0;
    let a2 = 0;
    let a3 = 0;
    let a4 = 0;
    let a5 = 0;
    let a6 = 0;
    let a7 = 0;

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0,
            inlateout("a1") a1,
            inlateout("a2") a2 => _,
            inlateout("a3") a3 => _,
            inlateout("a4") a4 => _,
            inlateout("a5") a5 => _,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };

    let result = a0;

    if result == SyscallResult::ThreadId as usize {
        Ok(a1.into())
    } else if result == SyscallResult::Error as usize {
        Err(a1.into())
    } else {
        Err(Error::InternalError)
    }
}

/// Adjusts the given `knob` limit to match the new value `new`. The current value must
/// match the `current` in order for this to take effect.
///
/// The new value is returned as a result of this call. If the call fails, then the old
/// value is returned. In either case, this function returns successfully.
///
/// An error is generated if the `knob` is not a valid limit, or if the call
/// would not succeed.
pub(crate) fn adjust_limit(knob: Limits, current: usize, new: usize) -> Result<usize, Error> {
    let mut a0 = Syscall::AdjustProcessLimit as usize;
    let mut a1 = knob as usize;
    let a2 = current;
    let a3 = new;
    let a4 = 0;
    let a5 = 0;
    let a6 = 0;
    let a7 = 0;

    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") a0,
            inlateout("a1") a1,
            inlateout("a2") a2 => _,
            inlateout("a3") a3 => _,
            inlateout("a4") a4 => _,
            inlateout("a5") a5 => _,
            inlateout("a6") a6 => _,
            inlateout("a7") a7 => _,
        )
    };

    let result = a0;

    if result == SyscallResult::Scalar2 as usize && a1 == knob as usize {
        Ok(a2)
    } else if result == SyscallResult::Scalar5 as usize && a1 == knob as usize {
        Ok(a1)
    } else if result == SyscallResult::Error as usize {
        Err(a1.into())
    } else {
        Err(Error::InternalError)
    }
}
