//! Functionality specific to the `x86_64-fortanix-unknown-sgx` target.
//!
//! This includes functions to deal with memory isolation, usercalls, and the
//! SGX instruction set.

#![deny(missing_docs, missing_debug_implementations)]
#![unstable(feature = "sgx_platform", issue = "56975")]

/// Low-level interfaces to usercalls. See the [ABI documentation] for more
/// information.
///
/// [ABI documentation]: https://docs.rs/fortanix-sgx-abi/
pub mod usercalls {
    pub use sys::abi::usercalls::*;

    /// Primitives for allocating memory in userspace as well as copying data
    /// to and from user memory.
    pub mod alloc {
        pub use sys::abi::usercalls::alloc;
    }

    /// Lowest-level interfaces to usercalls and usercall ABI type definitions.
    pub mod raw {
        use sys::abi::usercalls::raw::invoke_with_usercalls;
        pub use sys::abi::usercalls::raw::do_usercall;
        pub use sys::abi::usercalls::raw::{accept_stream, alloc, async_queues, bind_stream, close,
                                           connect_stream, exit, flush, free, insecure_time,
                                           launch_thread, read, read_alloc, send, wait, write};

        macro_rules! define_usercallnrs {
            ($(fn $f:ident($($n:ident: $t:ty),*) $(-> $r:ty)*; )*) => {
                /// Usercall numbers as per the ABI.
                #[repr(C)]
                #[unstable(feature = "sgx_platform", issue = "56975")]
                #[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
                #[allow(missing_docs)]
                pub enum UsercallNrs {
                    $($f,)*
                }
            };
        }
        invoke_with_usercalls!(define_usercallnrs);

        // fortanix-sgx-abi re-exports
        pub use sys::abi::usercalls::raw::{ByteBuffer, FifoDescriptor, Return, Usercall};
        pub use sys::abi::usercalls::raw::Error;
        pub use sys::abi::usercalls::raw::{EV_RETURNQ_NOT_EMPTY, EV_UNPARK, EV_USERCALLQ_NOT_FULL,
                                           FD_STDERR, FD_STDIN, FD_STDOUT, RESULT_SUCCESS,
                                           USERCALL_USER_DEFINED, WAIT_INDEFINITE, WAIT_NO};
        pub use sys::abi::usercalls::raw::{Fd, Result, Tcs};
    }
}

/// Functions for querying mapping information for pointers.
pub mod mem {
    pub use sys::abi::mem::*;
}
