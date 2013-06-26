// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use core::uint;
use core::vec;

/**
 * The Digest trait specifies an interface common to digest functions, such as SHA-1 and the SHA-2
 * family of digest functions.
 */
pub trait Digest {
    /**
     * Provide message data.
     *
     * # Arguments
     *
     * * input - A vector of message data
     */
    fn input(&mut self, input: &[u8]);

    /**
     * Retrieve the digest result. This method may be called multiple times.
     */
    fn result(&mut self, out: &mut [u8]);

    /**
     * Reset the digest. This method must be called after result() and before supplying more
     * data.
     */
    fn reset(&mut self);

    /**
     * Get the output size in bits.
     */
    fn output_bits(&self) -> uint;
}

fn to_hex(rr: &[u8]) -> ~str {
    let mut s = ~"";
    for rr.iter().advance() |b| {
        let hex = uint::to_str_radix(*b as uint, 16u);
        if hex.len() == 1 {
            s.push_char('0');
        }
        s.push_str(hex);
    }
    return s;
}

/// Contains utility methods for Digests.
/// FIXME: #7339: Convert to default methods when issues with them are resolved.
pub trait DigestUtil {
    /**
     * Convenience functon that feeds a string into a digest
     *
     * # Arguments
     *
     * * in The string to feed into the digest
     */
    fn input_str(&mut self, in: &str);

    /**
     * Convenience functon that retrieves the result of a digest as a
     * ~str in hexadecimal format.
     */
    fn result_str(&mut self) -> ~str;
}

impl<D: Digest> DigestUtil for D {
    fn input_str(&mut self, in: &str) {
        self.input(in.as_bytes());
    }

    fn result_str(&mut self) -> ~str {
        let mut buf = vec::from_elem((self.output_bits()+7)/8, 0u8);
        self.result(buf);
        return to_hex(buf);
    }
}
