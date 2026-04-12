// Ensure `ErrorKind` from `core` is gated behind `io_error_kind_in_core`
//@ edition:2024

use std::io::ErrorKind as ErrorKindFromStd;

use core::io::ErrorKind as ErrorKindFromCore;
//~^ ERROR use of unstable library feature `core_io`

// Asserting both ErrorKinds are the same.
const _: [ErrorKindFromCore; 1] = [ErrorKindFromStd::Other];

fn main() {}
