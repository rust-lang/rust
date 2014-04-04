// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A speedy hash algorithm for small keys. Note that although this can be far faster for small
//! keys than SipHash, this is not cryptographically secure and so should be avoided whenever it is
//! not absolutely certain that having hashes collide due to a malicious user is tolerable.
//!
//! Additionally, as this goes through the input one byte at a time, it loses its speed advantage
//! as the key gets larger.
//!
//! This uses FNV-1a hashing, as described here:
//! http://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function

use io::{Writer, IoResult};
use iter::Iterator;
use option::{None, Some};
use result::Ok;
use slice::ImmutableVector;

use super::{Hash, Hasher};

/// `FnvHasher` hashes data using the FNV-1a algorithm.
#[deriving(Clone)]
pub struct FnvHasher;

/// `FnvState` is the internal state of the FNV-1a algorithm while hashing a stream of bytes.
pub struct FnvState(u64);

impl Hasher<FnvState> for FnvHasher {
    fn hash<T: Hash<FnvState>>(&self, t: &T) -> u64 {
        let mut state = FnvState(0xcbf29ce484222325);
        t.hash(&mut state);
        let FnvState(ret) = state;
        return ret;
    }
}

impl Writer for FnvState {
    fn write(&mut self, bytes: &[u8]) -> IoResult<()> {
        let FnvState(mut hash) = *self;
        for byte in bytes.iter() {
            hash = hash ^ (*byte as u64);
            hash = hash * 0x100000001b3;
        }
        *self = FnvState(hash);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    extern crate test;
    use self::test::Bencher;

    use super::super::{Hash, Hasher};
    use super::FnvHasher;

    use str::StrSlice;

    #[bench]
    fn bench_str_under_8_bytes(b: &mut Bencher) {
        let s = "foo";
        b.iter(|| {
            assert_eq!(FnvHasher.hash(&s), 15929937188857697816);
        })
    }

    #[bench]
    fn bench_str_of_8_bytes(b: &mut Bencher) {
        let s = "foobar78";
        b.iter(|| {
            assert_eq!(FnvHasher.hash(&s), 5149438289095810352);
        })
    }

    #[bench]
    fn bench_str_over_8_bytes(b: &mut Bencher) {
        let s = "foobarbaz0";
        b.iter(|| {
            assert_eq!(FnvHasher.hash(&s), 4563734403032073248);
        })
    }

    #[bench]
    fn bench_long_str(b: &mut Bencher) {
        let s = "Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor \
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud \
exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute \
irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla \
pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui \
officia deserunt mollit anim id est laborum.";
        b.iter(|| {
            assert_eq!(FnvHasher.hash(&s), 18345634677153732273);
        })
    }

    #[bench]
    fn bench_u64(b: &mut Bencher) {
        let u = 16262950014981195938u64;
        b.iter(|| {
            assert_eq!(FnvHasher.hash(&u), 2430440085873150593);
        })
    }

    #[deriving(Hash)]
    struct Compound {
        x: u8,
        y: u64,
        z: ~str,
    }

    #[bench]
    fn bench_compound_1(b: &mut Bencher) {
        let compound = Compound {
            x: 1,
            y: 2,
            z: "foobarbaz".to_owned(),
        };
        b.iter(|| {
            assert_eq!(FnvHasher.hash(&compound), 3908687174522763645);
        })
    }
}
