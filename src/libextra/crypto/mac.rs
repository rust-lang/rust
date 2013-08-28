// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * The mac module defines the Message Authentication Code (Mac) trait.
 */

use cryptoutil::fixed_time_eq;

/**
 * The Mac trait defines methods for a Message Authentication function.
 */
pub trait Mac {
    /**
     * Process input data.
     *
     * # Arguments
     * * data - The input data to process.
     *
     */
    fn input(&mut self, data: &[u8]);

    /**
     * Reset the Mac state to begin processing another input stream.
     */
    fn reset(&mut self);

    /**
     * Obtain the result of a Mac computation as a MacResult.
     */
    fn result(&mut self) -> MacResult;

    /**
     * Obtain the result of a Mac computation as [u8]. This method should be used very carefully
     * since incorrect use of the Mac code could result in permitting a timing attack which defeats
     * the security provided by a Mac function.
     */
    fn raw_result(&mut self, output: &mut [u8]);

    /**
     * Get the size of the Mac code, in bytes.
     */
    fn output_bytes(&self) -> uint;
}

/**
 * A MacResult wraps a Mac code and provides a safe Eq implementation that runs in fixed time.
 */
pub struct MacResult {
    priv code: ~[u8]
}

impl MacResult {
    /**
     * Create a new MacResult.
     */
    #[experimental="EXPERIMENTAL. USE AT YOUR OWN RISK."]
    pub fn new(code: &[u8]) -> MacResult {
        return MacResult {
            code: code.to_owned()
        };
    }

    /**
     * Create a new MacResult taking ownership of the specified code value.
     */
    pub fn new_from_owned(code: ~[u8]) -> MacResult {
        return MacResult {
            code: code
        };
    }

    /**
     * Get the code value. Be very careful using this method, since incorrect use of the code value
     * may permit timing attacks which defeat the security provided by the Mac function.
     */
    pub fn code<'s>(&'s self) -> &'s [u8] {
        return self.code.as_slice();
    }
}

impl Eq for MacResult {
    fn eq(&self, x: &MacResult) -> bool {
        let lhs = self.code();
        let rhs = x.code();
        return fixed_time_eq(lhs, rhs);
    }
}
