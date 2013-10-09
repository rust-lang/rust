// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Interfaces to the operating system provided random number
//! generators.

use rand::Rng;
use ops::Drop;

#[cfg(unix)]
use rand::reader::ReaderRng;
#[cfg(unix)]
use rt::io::{file, Open, Read};

#[cfg(windows)]
use cast;
#[cfg(windows)]
use libc::{c_long, DWORD, BYTE};
#[cfg(windows)]
type HCRYPTPROV = c_long;
// the extern functions imported from the runtime on Windows are
// implemented so that they either succeed or abort(), so we can just
// assume they work when we call them.

/// A random number generator that retrieves randomness straight from
/// the operating system. On Unix-like systems this reads from
/// `/dev/urandom`, on Windows this uses `CryptGenRandom`.
///
/// This does not block.
#[cfg(unix)]
pub struct OSRng {
    priv inner: ReaderRng<file::FileStream>
}
/// A random number generator that retrieves randomness straight from
/// the operating system. On Unix-like systems this reads from
/// `/dev/urandom`, on Windows this uses `CryptGenRandom`.
///
/// This does not block.
#[cfg(windows)]
pub struct OSRng {
    priv hcryptprov: HCRYPTPROV
}

impl OSRng {
    /// Create a new `OSRng`.
    #[cfg(unix)]
    pub fn new() -> OSRng {
        let reader = file::open(& &"/dev/urandom", Open, Read).expect("Error opening /dev/urandom");
        let reader_rng = ReaderRng::new(reader);

        OSRng { inner: reader_rng }
    }

    /// Create a new `OSRng`.
    #[cfg(windows)]
    pub fn new() -> OSRng {
        externfn!(fn rust_win32_rand_acquire(phProv: *mut HCRYPTPROV))

        let mut hcp = 0;
        unsafe {rust_win32_rand_acquire(&mut hcp)};

        OSRng { hcryptprov: hcp }
    }
}

#[cfg(unix)]
impl Rng for OSRng {
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }
    fn fill_bytes(&mut self, v: &mut [u8]) {
        self.inner.fill_bytes(v)
    }
}

#[cfg(windows)]
impl Rng for OSRng {
    fn next_u32(&mut self) -> u32 {
        let mut v = [0u8, .. 4];
        self.fill_bytes(v);
        unsafe { cast::transmute(v) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut v = [0u8, .. 8];
        self.fill_bytes(v);
        unsafe { cast::transmute(v) }
    }
    fn fill_bytes(&mut self, v: &mut [u8]) {
        externfn!(fn rust_win32_rand_gen(hProv: HCRYPTPROV, dwLen: DWORD, pbBuffer: *mut BYTE))

        do v.as_mut_buf |ptr, len| {
            unsafe {rust_win32_rand_gen(self.hcryptprov, len as DWORD, ptr)}
        }
    }
}

impl Drop for OSRng {
    #[cfg(unix)]
    fn drop(&mut self) {
        // ensure that OSRng is not implicitly copyable on all
        // platforms, for consistency.
    }

    #[cfg(windows)]
    fn drop(&mut self) {
        externfn!(fn rust_win32_rand_release(hProv: HCRYPTPROV))

        unsafe {rust_win32_rand_release(self.hcryptprov)}
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_os_rng() {
        let mut r = OSRng::new();

        r.next_u32();
        r.next_u64();

        let mut v = [0u8, .. 1000];
        r.fill_bytes(v);
    }

    #[test]
    fn test_os_rng_tasks() {
        use task;
        use comm;
        use comm::{GenericChan, GenericPort};
        use option::{None, Some};
        use iter::{Iterator, range};
        use vec::{ImmutableVector, OwnedVector};

        let mut chans = ~[];
        for _ in range(0, 20) {
            let (p, c) = comm::stream();
            chans.push(c);
            do task::spawn_with(p) |p| {
                // wait until all the tasks are ready to go.
                p.recv();

                // deschedule to attempt to interleave things as much
                // as possible (XXX: is this a good test?)
                let mut r = OSRng::new();
                task::deschedule();
                let mut v = [0u8, .. 1000];

                for _ in range(0, 100) {
                    r.next_u32();
                    task::deschedule();
                    r.next_u64();
                    task::deschedule();
                    r.fill_bytes(v);
                    task::deschedule();
                }
            }
        }

        // start all the tasks
        for c in chans.iter() {
            c.send(())
        }
    }
}
