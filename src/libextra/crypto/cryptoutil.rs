// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::num::One;
use std::vec::bytes::{MutableByteVector, copy_memory};


/// Write a u64 into a vector, which must be 8 bytes long. The value is written in big-endian
/// format.
pub fn write_u64_be(dst: &mut[u8], input: u64) {
    use std::cast::transmute;
    use std::unstable::intrinsics::to_be64;
    assert!(dst.len() == 8);
    unsafe {
        let x: *mut i64 = transmute(dst.unsafe_mut_ref(0));
        *x = to_be64(input as i64);
    }
}

/// Write a u32 into a vector, which must be 4 bytes long. The value is written in big-endian
/// format.
pub fn write_u32_be(dst: &mut[u8], input: u32) {
    use std::cast::transmute;
    use std::unstable::intrinsics::to_be32;
    assert!(dst.len() == 4);
    unsafe {
        let x: *mut i32 = transmute(dst.unsafe_mut_ref(0));
        *x = to_be32(input as i32);
    }
}

/// Read a vector of bytes into a vector of u64s. The values are read in big-endian format.
pub fn read_u64v_be(dst: &mut[u64], input: &[u8]) {
    use std::cast::transmute;
    use std::unstable::intrinsics::to_be64;
    assert!(dst.len() * 8 == input.len());
    unsafe {
        let mut x: *mut i64 = transmute(dst.unsafe_mut_ref(0));
        let mut y: *i64 = transmute(input.unsafe_ref(0));
        do dst.len().times() {
            *x = to_be64(*y);
            x = x.offset(1);
            y = y.offset(1);
        }
    }
}

/// Read a vector of bytes into a vector of u32s. The values are read in big-endian format.
pub fn read_u32v_be(dst: &mut[u32], input: &[u8]) {
    use std::cast::transmute;
    use std::unstable::intrinsics::to_be32;
    assert!(dst.len() * 4 == input.len());
    unsafe {
        let mut x: *mut i32 = transmute(dst.unsafe_mut_ref(0));
        let mut y: *i32 = transmute(input.unsafe_ref(0));
        do dst.len().times() {
            *x = to_be32(*y);
            x = x.offset(1);
            y = y.offset(1);
        }
    }
}


/// Returns true if adding the two parameters will result in integer overflow
pub fn will_add_overflow<T: Int + Unsigned>(x: T, y: T) -> bool {
    // This doesn't handle negative values! Don't copy this code elsewhere without considering if
    // negative values are important to you!
    let max: T = Bounded::max_value();
    return x > max - y;
}

/// Shifts the second parameter and then adds it to the first. fails!() if there would be unsigned
/// integer overflow.
pub fn shift_add_check_overflow<T: Int + Unsigned + Clone>(x: T, mut y: T, shift: T) -> T {
    if y.leading_zeros() < shift {
        fail!("Could not add values - integer overflow.");
    }
    y = y << shift;

    if will_add_overflow(x.clone(), y.clone()) {
        fail!("Could not add values - integer overflow.");
    }

    return x + y;
}

/// Shifts the second parameter and then adds it to the first, which is a tuple where the first
/// element is the high order value. fails!() if there would be unsigned integer overflow.
pub fn shift_add_check_overflow_tuple
        <T: Int + Unsigned + Clone>
        (x: (T, T), mut y: T, shift: T) -> (T, T) {
    if y.leading_zeros() < shift {
        fail!("Could not add values - integer overflow.");
    }
    y = y << shift;

    match x {
        (hi, low) => {
            let one: T = One::one();
            if will_add_overflow(low.clone(), y.clone()) {
                if will_add_overflow(hi.clone(), one.clone()) {
                    fail!("Could not add values - integer overflow.");
                } else {
                    return (hi + one, low + y);
                }
            } else {
                return (hi, low + y);
            }
        }
    }
}


/// A FixedBuffer, likes its name implies, is a fixed size buffer. When the buffer becomes full, it
/// must be processed. The input() method takes care of processing and then clearing the buffer
/// automatically. However, other methods do not and require the caller to process the buffer. Any
/// method that modifies the buffer directory or provides the caller with bytes that can be modifies
/// results in those bytes being marked as used by the buffer.
pub trait FixedBuffer {
    /// Input a vector of bytes. If the buffer becomes full, proccess it with the provided
    /// function and then clear the buffer.
    fn input(&mut self, input: &[u8], func: &fn(&[u8]));

    /// Reset the buffer.
    fn reset(&mut self);

    /// Zero the buffer up until the specified index. The buffer position currently must not be
    /// greater than that index.
    fn zero_until(&mut self, idx: uint);

    /// Get a slice of the buffer of the specified size. There must be at least that many bytes
    /// remaining in the buffer.
    fn next<'s>(&'s mut self, len: uint) -> &'s mut [u8];

    /// Get the current buffer. The buffer must already be full. This clears the buffer as well.
    fn full_buffer<'s>(&'s mut self) -> &'s [u8];

    /// Get the current position of the buffer.
    fn position(&self) -> uint;

    /// Get the number of bytes remaining in the buffer until it is full.
    fn remaining(&self) -> uint;

    /// Get the size of the buffer
    fn size(&self) -> uint;
}

macro_rules! impl_fixed_buffer( ($name:ident, $size:expr) => (
    impl FixedBuffer for $name {
        fn input(&mut self, input: &[u8], func: &fn(&[u8])) {
            let mut i = 0;

            // FIXME: #6304 - This local variable shouldn't be necessary.
            let size = $size;

            // If there is already data in the buffer, copy as much as we can into it and process
            // the data if the buffer becomes full.
            if self.buffer_idx != 0 {
                let buffer_remaining = size - self.buffer_idx;
                if input.len() >= buffer_remaining {
                        copy_memory(
                            self.buffer.mut_slice(self.buffer_idx, size),
                            input.slice_to(buffer_remaining),
                            buffer_remaining);
                    self.buffer_idx = 0;
                    func(self.buffer);
                    i += buffer_remaining;
                } else {
                    copy_memory(
                        self.buffer.mut_slice(self.buffer_idx, self.buffer_idx + input.len()),
                        input,
                        input.len());
                    self.buffer_idx += input.len();
                    return;
                }
            }

            // While we have at least a full buffer size chunks's worth of data, process that data
            // without copying it into the buffer
            while input.len() - i >= size {
                func(input.slice(i, i + size));
                i += size;
            }

            // Copy any input data into the buffer. At this point in the method, the ammount of
            // data left in the input vector will be less than the buffer size and the buffer will
            // be empty.
            let input_remaining = input.len() - i;
            copy_memory(
                self.buffer.mut_slice(0, input_remaining),
                input.slice_from(i),
                input.len() - i);
            self.buffer_idx += input_remaining;
        }

        fn reset(&mut self) {
            self.buffer_idx = 0;
        }

        fn zero_until(&mut self, idx: uint) {
            assert!(idx >= self.buffer_idx);
            self.buffer.mut_slice(self.buffer_idx, idx).set_memory(0);
            self.buffer_idx = idx;
        }

        fn next<'s>(&'s mut self, len: uint) -> &'s mut [u8] {
            self.buffer_idx += len;
            return self.buffer.mut_slice(self.buffer_idx - len, self.buffer_idx);
        }

        fn full_buffer<'s>(&'s mut self) -> &'s [u8] {
            assert!(self.buffer_idx == $size);
            self.buffer_idx = 0;
            return self.buffer.slice_to($size);
        }

        fn position(&self) -> uint { self.buffer_idx }

        fn remaining(&self) -> uint { $size - self.buffer_idx }

        fn size(&self) -> uint { $size }
    }
))


/// A fixed size buffer of 64 bytes useful for cryptographic operations.
pub struct FixedBuffer64 {
    priv buffer: [u8, ..64],
    priv buffer_idx: uint,
}

impl FixedBuffer64 {
    /// Create a new buffer
    pub fn new() -> FixedBuffer64 {
        return FixedBuffer64 {
            buffer: [0u8, ..64],
            buffer_idx: 0
        };
    }
}

impl_fixed_buffer!(FixedBuffer64, 64)

/// A fixed size buffer of 128 bytes useful for cryptographic operations.
pub struct FixedBuffer128 {
    priv buffer: [u8, ..128],
    priv buffer_idx: uint,
}

impl FixedBuffer128 {
    /// Create a new buffer
    pub fn new() -> FixedBuffer128 {
        return FixedBuffer128 {
            buffer: [0u8, ..128],
            buffer_idx: 0
        };
    }
}

impl_fixed_buffer!(FixedBuffer128, 128)


/// The StandardPadding trait adds a method useful for various hash algorithms to a FixedBuffer
/// struct.
pub trait StandardPadding {
    /// Add standard padding to the buffer. The buffer must not be full when this method is called
    /// and is guaranteed to have exactly rem remaining bytes when it returns. If there are not at
    /// least rem bytes available, the buffer will be zero padded, processed, cleared, and then
    /// filled with zeros again until only rem bytes are remaining.
    fn standard_padding(&mut self, rem: uint, func: &fn(&[u8]));
}

impl <T: FixedBuffer> StandardPadding for T {
    fn standard_padding(&mut self, rem: uint, func: &fn(&[u8])) {
        let size = self.size();

        self.next(1)[0] = 128;

        if self.remaining() < rem {
            self.zero_until(size);
            func(self.full_buffer());
        }

        self.zero_until(size - rem);
    }
}


#[cfg(test)]
mod test {
    use std::rand::IsaacRng;
    use std::rand::RngUtil;
    use std::vec;

    use digest::Digest;

    /// Feed 1,000,000 'a's into the digest with varying input sizes and check that the result is
    /// correct.
    pub fn test_digest_1million_random<D: Digest>(digest: &mut D, blocksize: uint, expected: &str) {
        let total_size = 1000000;
        let buffer = vec::from_elem(blocksize * 2, 'a' as u8);
        let mut rng = IsaacRng::new_unseeded();
        let mut count = 0;

        digest.reset();

        while count < total_size {
            let next: uint = rng.gen_uint_range(0, 2 * blocksize + 1);
            let remaining = total_size - count;
            let size = if next > remaining { remaining } else { next };
            digest.input(buffer.slice_to(size));
            count += size;
        }

        let result_str = digest.result_str();

        assert!(expected == result_str);
    }
}
