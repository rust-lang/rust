// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fortanix_sgx_abi::Tcs;

/// Get the ID for the current thread. The ID is guaranteed to be unique among
/// all currently running threads in the enclave, and it is guaranteed to be
/// constant for the lifetime of the thread. More specifically for SGX, there
/// is a one-to-one correspondence of the ID to the address of the TCS.
pub fn current() -> Tcs {
    extern "C" { fn get_tcs_addr() -> Tcs; }
    unsafe { get_tcs_addr() }
}
