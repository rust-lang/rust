#![stable(feature = "rust1", since = "1.0.0")]

use hermit_abi as abi;

#[stable(feature = "rust1", since = "1.0.0")]
pub type RawFd = abi::FileDescriptor;
