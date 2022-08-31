//! Functionality specific to the `x86_64-fortanix-unknown-sgx` target.
//!
//! This includes functions to deal with memory isolation, usercalls, and the
//! SGX instruction set.

#![deny(missing_docs)]
#![unstable(feature = "sgx_platform", issue = "56975")]

/// Low-level interfaces to usercalls. See the [ABI documentation] for more
/// information.
///
/// [ABI documentation]: https://docs.rs/fortanix-sgx-abi/
pub mod usercalls {
    pub use crate::sys::abi::usercalls::*;

    /// Primitives for allocating memory in userspace as well as copying data
    /// to and from user memory.
    pub mod alloc {
        pub use crate::sys::abi::usercalls::alloc::*;
    }

    /// Lowest-level interfaces to usercalls and usercall ABI type definitions.
    pub mod raw {
        pub use crate::sys::abi::usercalls::raw::{
            accept_stream, alloc, async_queues, bind_stream, close, connect_stream, exit, flush,
            free, insecure_time, launch_thread, read, read_alloc, send, wait, write,
        };
        pub use crate::sys::abi::usercalls::raw::{do_usercall, Usercalls as UsercallNrs};
        pub use crate::sys::abi::usercalls::raw::{Register, RegisterArgument, ReturnValue};

        // fortanix-sgx-abi re-exports
        pub use crate::sys::abi::usercalls::raw::Error;
        pub use crate::sys::abi::usercalls::raw::{
            ByteBuffer, Cancel, FifoDescriptor, Return, Usercall,
        };
        pub use crate::sys::abi::usercalls::raw::{Fd, Result, Tcs};
        pub use crate::sys::abi::usercalls::raw::{
            EV_RETURNQ_NOT_EMPTY, EV_UNPARK, EV_USERCALLQ_NOT_FULL, FD_STDERR, FD_STDIN, FD_STDOUT,
            RESULT_SUCCESS, USERCALL_USER_DEFINED, WAIT_INDEFINITE, WAIT_NO,
        };
    }
}

/// Functions for querying mapping information for pointers.
pub mod mem {
    pub use crate::sys::abi::mem::*;
}

pub mod arch;
pub mod ffi;
pub mod io;

/// Functions for querying thread-related information.
pub mod thread {
    pub use crate::sys::abi::thread::current;
}
