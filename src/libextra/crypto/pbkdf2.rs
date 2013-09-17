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
 * This module implements the PBKDF2 Key Derivation Function as specified by
 * http://tools.ietf.org/html/rfc2898.
 */

use std::rand::{IsaacRng, RngUtil};
use std::vec;
use std::vec::MutableCloneableVector;

use base64;
use base64::{FromBase64, ToBase64};
use cryptoutil::{read_u32_be, write_u32_be, fixed_time_eq};
use hmac::Hmac;
use mac::Mac;
use sha2::Sha256;

// Calculate a block of the output of size equal to the output_bytes of the underlying Mac function
// mac - The Mac function to use
// salt - the salt value to use
// c - the iteration count
// idx - the 1 based index of the block
// scratch - a temporary variable the same length as the block
// block - the block of the output to calculate
fn calculate_block<M: Mac>(
        mac: &mut M,
        salt: &[u8],
        c: u32,
        idx: u32,
        scratch: &mut [u8],
        block: &mut [u8]) {
    // Perform the 1st iteration. The output goes directly into block
    mac.input(salt);
    let mut idx_buf = [0u8, ..4];
    write_u32_be(idx_buf, idx);
    mac.input(idx_buf);
    mac.raw_result(block);
    mac.reset();

    // Perform the 2nd iteration. The input comes from block and is output into scratch. scratch is
    // then exclusive-or added into block. After all this, the input to the next step is now in
    // scratch and block is left to just accumulate the exclusive-of sum of remaining iterations.
    if c > 1 {
        mac.input(block);
        mac.raw_result(scratch);
        mac.reset();
        for (output, &input) in block.mut_iter().zip(scratch.iter()) {
            *output ^= input;
        }
    }

    // Perform all remaining iterations
    for _ in range(2, c) {
        mac.input(scratch);
        mac.raw_result(scratch);
        mac.reset();
        for (output, &input) in block.mut_iter().zip(scratch.iter()) {
            *output ^= input;
        }
    }
}

/**
 * Execute the PBKDF2 Key Derivation Function. The Scrypt Key Derivation Function generally provides
 * better security, so, applications that do not have a requirement to use PBKDF2 specifically
 * should consider using that function instead.
 *
 * # Arguments
 * * mac - The Pseudo Random Function to use.
 * * salt - The salt value to use.
 * * c - The iteration count. Users should carefully determine this value as it is the primary
 *       factor in determining the security of the derived key.
 * * output - The output buffer to fill with the derived key value.
 *
 */
#[experimental="EXPERIMENTAL. USE AT YOUR OWN RISK."]
pub fn pbkdf2<M: Mac>(mac: &mut M, salt: &[u8], c: u32, output: &mut [u8]) {
    assert!(c > 0);

    let os = mac.output_bytes();

    // A temporary storage array needed by calculate_block. This is really only necessary if c > 1.
    // Most users of pbkdf2 should use a value much larger than 1, so, this allocation should almost
    // always be necessary. A big exception is Scrypt. However, this allocation is unlikely to be
    // the bottleneck in Scrypt performance.
    let mut scratch = vec::from_elem(os, 0u8);

    let mut idx: u32 = 0;

    for chunk in output.mut_chunk_iter(os) {
        if idx == Bounded::max_value() {
            fail!("PBKDF2 size limit exceeded.");
        } else {
            // The block index starts at 1. So, this is supposed to run on the first execution.
            idx += 1;
        }
        if chunk.len() == os {
            calculate_block(mac, salt, c, idx, scratch, chunk);
        } else {
            let mut tmp = vec::from_elem(os, 0u8);
            calculate_block(mac, salt, c, idx, scratch, tmp);
            chunk.copy_from(tmp);
        }
    }
}

/**
 * pbkdf2_simple is a helper function that should be sufficient for the majority of cases where
 * an application needs to use PBKDF2 to hash a password for storage. The result is a ~str that
 * contains the parameters used as part of its encoding. The pbkdf2_check function may be used on
 * a password to check if it is equal to a hashed value.
 *
 * # Format
 *
 * The format of the output is a modified version of the Modular Crypt Format that encodes algorithm
 * used and iteration count. The format is indicated as "rpbkdf2" which is short for "Rust PBKF2
 * format."
 *
 * $rpbkdf2$0$<base64(c)>$<base64(salt)>$<based64(hash)>$
 *
 * # Arguments
 *
 * * password - The password to process as a str
 * * c - The iteration count
 *
 */
#[experimental="EXPERIMENTAL. USE AT YOUR OWN RISK."]
pub fn pbkdf2_simple(password: &str, c: u32) -> ~str {
    let mut rng = IsaacRng::new();

    // 128-bit salt
    let salt = rng.gen_bytes(16);

    // 256-bit derived key
    let mut dk = [0u8, ..32];

    let mut mac = Hmac::new(Sha256::new(), password.as_bytes());

    pbkdf2(&mut mac, salt, c, dk);

    let mut result = ~"$rpbkdf2$0$";
    let mut tmp = [0u8, ..4];
    write_u32_be(tmp, c);
    result.push_str(tmp.to_base64(base64::STANDARD));
    result.push_char('$');
    result.push_str(salt.to_base64(base64::STANDARD));
    result.push_char('$');
    result.push_str(dk.to_base64(base64::STANDARD));
    result.push_char('$');

    return result;
}

/**
 * pbkdf2_check compares a password against the result of a previous call to pbkdf2_simple and
 * returns true if the passed in password hashes to the same value.
 *
 * # Arguments
 *
 * * password - The password to process as a str
 * * hashed_value - A string representing a hashed password returned by pbkdf2_simple()
 *
 */
#[experimental="EXPERIMENTAL. USE AT YOUR OWN RISK."]
pub fn pbkdf2_check(password: &str, hashed_value: &str) -> Result<bool, &'static str> {
    static ERR_STR: &'static str = "Hash is not in Rust PBKDF2 format.";

    let mut iter = hashed_value.split_iter('$');

    // Check that there are no characters before the first "$"
    match iter.next() {
        Some(x) => if x != "" { return Err(ERR_STR); },
        None => return Err(ERR_STR)
    }

    // Check the name
    match iter.next() {
        Some(t) => if t != "rpbkdf2" { return Err(ERR_STR); },
        None => return Err(ERR_STR)
    }

    // Parse format - currenlty only version 0 is supported
    match iter.next() {
        Some(fstr) => {
            match fstr {
                "0" => { }
                _ => return Err(ERR_STR)
            }
        }
        None => return Err(ERR_STR)
    }

    // Parse the iteration count
    let c = match iter.next() {
        Some(pstr) => match pstr.from_base64() {
            Ok(pvec) => {
                if pvec.len() != 4 { return Err(ERR_STR); }
                read_u32_be(pvec)
            }
            Err(_) => return Err(ERR_STR)
        },
        None => return Err(ERR_STR)
    };

    // Salt
    let salt = match iter.next() {
        Some(sstr) => match sstr.from_base64() {
            Ok(salt) => salt,
            Err(_) => return Err(ERR_STR)
        },
        None => return Err(ERR_STR)
    };

    // Hashed value
    let hash = match iter.next() {
        Some(hstr) => match hstr.from_base64() {
            Ok(hash) => hash,
            Err(_) => return Err(ERR_STR)
        },
        None => return Err(ERR_STR)
    };

    // Make sure that the input ends with a "$"
    match iter.next() {
        Some(x) => if x != "" { return Err(ERR_STR); },
        None => return Err(ERR_STR)
    }

    // Make sure there is no trailing data after the final "$"
    match iter.next() {
        Some(_) => return Err(ERR_STR),
        None => { }
    }

    let mut mac = Hmac::new(Sha256::new(), password.as_bytes());

    let mut output = vec::from_elem(hash.len(), 0u8);
    pbkdf2(&mut mac, salt, c, output);

    // Be careful here - its important that the comparison be done using a fixed time equality
    // check. Otherwise an adversary that can measure how long this step takes can learn about the
    // hashed value which would allow them to mount an offline brute force attack against the
    // hashed password.
    return Ok(fixed_time_eq(output, hash));
}

#[cfg(test)]
mod test {
    use std::vec;

    use pbkdf2::{pbkdf2, pbkdf2_simple, pbkdf2_check};
    use hmac::Hmac;
    use sha1::Sha1;

    struct Test {
        password: ~[u8],
        salt: ~[u8],
        c: u32,
        expected: ~[u8]
    }

    // Test vectors from http://tools.ietf.org/html/rfc6070. The 4th test vector is omitted because
    // it takes too long to run.

    fn tests() -> ~[Test] {
        return ~[
            Test {
                password: "password".as_bytes().to_owned(),
                salt: "salt".as_bytes().to_owned(),
                c: 1,
                expected: ~[
                    0x0c, 0x60, 0xc8, 0x0f, 0x96, 0x1f, 0x0e, 0x71,
                    0xf3, 0xa9, 0xb5, 0x24, 0xaf, 0x60, 0x12, 0x06,
                    0x2f, 0xe0, 0x37, 0xa6 ]
            },
            Test {
                password: "password".as_bytes().to_owned(),
                salt: "salt".as_bytes().to_owned(),
                c: 2,
                expected: ~[
                    0xea, 0x6c, 0x01, 0x4d, 0xc7, 0x2d, 0x6f, 0x8c,
                    0xcd, 0x1e, 0xd9, 0x2a, 0xce, 0x1d, 0x41, 0xf0,
                    0xd8, 0xde, 0x89, 0x57 ]
            },
            Test {
                password: "password".as_bytes().to_owned(),
                salt: "salt".as_bytes().to_owned(),
                c: 4096,
                expected: ~[
                    0x4b, 0x00, 0x79, 0x01, 0xb7, 0x65, 0x48, 0x9a,
                    0xbe, 0xad, 0x49, 0xd9, 0x26, 0xf7, 0x21, 0xd0,
                    0x65, 0xa4, 0x29, 0xc1 ]
            },
            Test {
                password: "passwordPASSWORDpassword".as_bytes().to_owned(),
                salt: "saltSALTsaltSALTsaltSALTsaltSALTsalt".as_bytes().to_owned(),
                c: 4096,
                expected: ~[
                    0x3d, 0x2e, 0xec, 0x4f, 0xe4, 0x1c, 0x84, 0x9b,
                    0x80, 0xc8, 0xd8, 0x36, 0x62, 0xc0, 0xe4, 0x4a,
                    0x8b, 0x29, 0x1a, 0x96, 0x4c, 0xf2, 0xf0, 0x70, 0x38 ]
            },
            Test {
                password: ~[112, 97, 115, 115, 0, 119, 111, 114, 100],
                salt: ~[115, 97, 0, 108, 116],
                c: 4096,
                expected: ~[
                    0x56, 0xfa, 0x6a, 0xa7, 0x55, 0x48, 0x09, 0x9d,
                    0xcc, 0x37, 0xd7, 0xf0, 0x34, 0x25, 0xe0, 0xc3 ]
            }
        ];
    }

    #[test]
    fn test_pbkdf2() {
        let tests = tests();
        for t in tests.iter() {
            let mut mac = Hmac::new(Sha1::new(), t.password);
            let mut result = vec::from_elem(t.expected.len(), 0u8);
            pbkdf2(&mut mac, t.salt, t.c, result);
            assert!(result == t.expected);
        }
    }

    #[test]
    fn test_pbkdf2_simple() {
        let password = "password";

        let out1 = pbkdf2_simple(password, 1024);
        let out2 = pbkdf2_simple(password, 1024);

        // This just makes sure that a salt is being applied. It doesn't verify that that salt is
        // cryptographically strong, however.
        assert!(out1 != out2);

        match pbkdf2_check(password, out1) {
            Ok(r) => assert!(r),
            Err(_) => fail!()
        }
        match pbkdf2_check(password, out2) {
            Ok(r) => assert!(r),
            Err(_) => fail!()
        }

        match pbkdf2_check("wrong", out1) {
            Ok(r) => assert!(!r),
            Err(_) => fail!()
        }
        match pbkdf2_check("wrong", out2) {
            Ok(r) => assert!(!r),
            Err(_) => fail!()
        }
    }
}
