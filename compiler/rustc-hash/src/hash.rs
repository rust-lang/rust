// copied over from the museair crate for benchmarking purposes only!
// some items that are not needed for the compiler have been removed

/*
 * Copyright (c) 2024 K--Aethiax
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Alternatively, the contents of this file may be used under the terms of
 * the MIT license as described below.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#![forbid(unsafe_code)]

/// Computes the 64-bit MuseAir hash for a byte slice. *(BFast variant)*
#[inline]
pub const fn hash(bytes: &[u8], seed: u64) -> u64 {
    impls::hash_64::<true>(bytes, seed)
}

/// Computes the 128-bit MuseAir hash for a byte slice. *(BFast variant)*
#[inline]
pub const fn hash_128(bytes: &[u8], seed: u64) -> u128 {
    impls::hash_128::<true>(bytes, seed)
}

/// An incremental [`Hasher`] instance that uses the MuseAir hashing algorithm. *(BFast variant)*
///
/// Note that incrementally hashing small keys will be significantly slower than one-shot hashing.
///
/// For better hashmap performance on small keys and enhanced HashDoS resistance,
/// consider using [`musemap`](https://crates.io/crates/musemap).
///
/// [`Hasher`]: core::hash::Hasher
pub type Hasher = impls::IncrementalHasher<true>;

/// A [`BuildHasher`] for the MuseAir incremental [`Hasher`] that always has the same fixed seed.
/// *(BFast variant)*
///
/// Note that incrementally hashing small keys will be significantly slower than one-shot hashing.
///
/// For better hashmap performance on small keys and enhanced HashDoS resistance,
/// consider using [`musemap`](https://crates.io/crates/musemap).
///
/// [`BuildHasher`]: core::hash::BuildHasher
pub type FixedState = impls::FixedState<true>;

//------------------------------------------------------------------------------

type State = [u64; 6];
type Chunk = [u8; 96];

/// `AiryAi(0)` fractional part calculated by Y-Cruncher.
const CONSTANT: [u64; 7] = [
    0x5ae31e589c56e17a,
    0x96d7bb04e64f6da9,
    0x7ab1006b26f9eb64,
    0x21233394220b8457,
    0x047cb9557c9f3b43,
    0xd24f2590c0bcee28,
    0x33ea8f71bb6016d8,
];

/// Lower 64-bit, then upper 64-bit.
#[inline(always)]
const fn wmul(a: u64, b: u64) -> (u64, u64) {
    u128_to_u64s(a as u128 * b as u128)
}

/// Lower 64-bit, then upper 64-bit.
#[inline(always)]
const fn u128_to_u64s(x: u128) -> (u64, u64) {
    (x as u64, (x >> 64) as u64)
}
/// Lower 64-bit, then upper 64-bit.
#[inline(always)]
const fn u64s_to_u128(lo: u64, hi: u64) -> u128 {
    ((hi as u128) << 64) | lo as u128
}

#[cold]
#[inline(always)]
const fn cold_path() {}

#[inline(always)]
const fn likely(cond: bool) -> bool {
    if !cond {
        cold_path();
    }
    cond
}
#[inline(always)]
const fn unlikely(cond: bool) -> bool {
    if cond {
        cold_path();
    }
    cond
}

macro_rules! u64 {
    ($n:literal) => {
        $n * 8
    };
}

macro_rules! min {
    ( $left:expr, $right:expr $(,)? ) => {
        match ($left, $right) {
            (left_val, right_val) => {
                if left_val < right_val {
                    left_val
                } else {
                    right_val
                }
            }
        }
    };
}

#[inline(always)]
const fn seed_state(seed: u64) -> State {
    [
        CONSTANT[0].wrapping_add(seed),
        CONSTANT[1].wrapping_sub(seed),
        CONSTANT[2] ^ seed,
        CONSTANT[3].wrapping_add(seed),
        CONSTANT[4].wrapping_sub(seed),
        CONSTANT[5] ^ seed,
    ]
}

//------------------------------------------------------------------------------

#[inline(always)]
const fn read_u32(bytes: &[u8], offset: usize) -> u64 {
    u32::from_le_bytes(*bytes.split_at(offset).1.first_chunk().unwrap()) as u64
}
#[inline(always)]
const fn read_u32_r(bytes: &[u8], offset_r: usize) -> u64 {
    u32::from_le_bytes(*bytes.split_at(bytes.len() - offset_r).0.last_chunk().unwrap()) as u64
}

#[inline(always)]
const fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(*bytes.split_at(offset).1.first_chunk().unwrap())
}
#[inline(always)]
const fn read_u64_r(bytes: &[u8], offset_r: usize) -> u64 {
    u64::from_le_bytes(*bytes.split_at(bytes.len() - offset_r).0.last_chunk().unwrap())
}

#[inline(always)]
const fn read_short(bytes: &[u8]) -> (u64, u64) {
    debug_assert!(bytes.len() <= u64!(2));

    let len = bytes.len();
    if len >= 4 {
        let off = (len & 24) >> (len >> 3); // len >= 8 ? 4 : 0
        let head = read_u32(bytes, 0);
        let head_off = read_u32(bytes, off);
        let tail = read_u32_r(bytes, 0);
        let tail_off = read_u32_r(bytes, off);

        (head << 32 | tail, head_off << 32 | tail_off)
    } else if len > 0 {
        // MSB <-> LSB
        // [0] [0] [0] @ len == 1 (0b01)
        // [0] [1] [1] @ len == 2 (0b10)
        // [0] [1] [2] @ len == 3 (0b11)
        let fst = bytes[0] as u64;
        let snd = bytes[len >> 1] as u64;
        let thd = bytes[len - 1] as u64;

        (fst << 48 | snd << 24 | thd, 0)
    } else {
        (0, 0)
    }
}

//------------------------------------------------------------------------------

#[inline(always)]
const fn hash_short_64<const BFAST: bool>(bytes: &[u8], seed: u64) -> u64 {
    let (mut i, mut j);
    (i, j) = hash_short_common(bytes, seed);
    if !BFAST {
        let (lo0, hi0) = wmul(i ^ CONSTANT[2], j ^ CONSTANT[3]);
        let (lo1, hi1) = wmul(i ^ CONSTANT[4], j ^ CONSTANT[5]);
        (i, j) = (lo0 ^ hi1, lo1 ^ hi0);
        let (lo2, hi2) = wmul(i, j);
        i ^ j ^ lo2 ^ hi2
    } else {
        (i, j) = wmul(i ^ CONSTANT[2], j ^ CONSTANT[3]);
        (i, j) = wmul(i ^ CONSTANT[4], j ^ CONSTANT[5]);
        i ^ j
    }
}

#[inline(always)]
const fn hash_short_128(bytes: &[u8], seed: u64) -> u128 {
    let (mut i, mut j);
    (i, j) = hash_short_common(bytes, seed);
    let (lo0, hi0) = wmul(i, j);
    let (lo1, hi1) = wmul(i ^ CONSTANT[2], j ^ CONSTANT[3]);
    (i, j) = (lo0 ^ hi1, lo1 ^ hi0);
    let (lo0, hi0) = wmul(i, j);
    let (lo1, hi1) = wmul(i ^ CONSTANT[4], j ^ CONSTANT[5]);
    (i, j) = (lo0 ^ hi1, lo1 ^ hi0);

    u64s_to_u128(i, j)
}

#[inline(always)]
const fn hash_short_common(bytes: &[u8], seed: u64) -> (u64, u64) {
    debug_assert!(bytes.len() <= u64!(4));

    let len = bytes.len();
    let len_ = bytes.len() as u64;
    let (lo2, hi2) = wmul(seed ^ CONSTANT[0], len_ ^ CONSTANT[1]);

    let (mut i, mut j) = read_short(bytes.split_at(min!(u64!(2), len)).0);
    i ^= len_ ^ lo2;
    j ^= seed ^ hi2;

    if unlikely(len > u64!(2)) {
        let (u, v) = read_short(bytes.split_at(u64!(2)).1);
        let (lo0, hi0) = wmul(CONSTANT[2], CONSTANT[3] ^ u);
        let (lo1, hi1) = wmul(CONSTANT[4], CONSTANT[5] ^ v);
        i ^= lo0 ^ hi1;
        j ^= lo1 ^ hi0;
    }

    (i, j)
}

//------------------------------------------------------------------------------

#[inline(always)]
const fn epilogue_64((i, j, k): (u64, u64, u64)) -> u64 {
    let (lo0, hi0) = wmul(i, j);
    let (lo1, hi1) = wmul(j, k);
    let (lo2, hi2) = wmul(k, i);
    (lo0 ^ hi2).wrapping_add(lo1 ^ hi0).wrapping_add(lo2 ^ hi1)
}

#[inline(always)]
const fn epilogue_128((i, j, k): (u64, u64, u64)) -> u128 {
    let (lo0, hi0) = wmul(i, j);
    let (lo1, hi1) = wmul(j, k);
    let (lo2, hi2) = wmul(k, i);
    u64s_to_u128(lo0 ^ lo1 ^ hi2, hi0 ^ hi1 ^ lo2)
}

//------------------------------------------------------------------------------

#[inline(never)]
const fn hash_loong_64<const BFAST: bool>(bytes: &[u8], seed: u64) -> u64 {
    epilogue_64(hash_loong_common::<BFAST>(bytes, seed))
}

#[inline(never)]
const fn hash_loong_128<const BFAST: bool>(bytes: &[u8], seed: u64) -> u128 {
    epilogue_128(hash_loong_common::<BFAST>(bytes, seed))
}

#[inline(always)]
const fn hash_loong_common<const BFAST: bool>(bytes: &[u8], seed: u64) -> (u64, u64, u64) {
    debug_assert!(bytes.len() > u64!(4));

    let mut remainder = bytes;
    let mut state = seed_state(seed);
    let [mut lo0, mut lo1, mut lo2, mut lo3, mut lo4, mut lo5];
    let [mut hi0, mut hi1, mut hi2, mut hi3, mut hi4, mut hi5];

    if unlikely(remainder.len() > u64!(12)) {
        lo5 = CONSTANT[6];

        while let Some((chunk, rest)) = remainder.split_first_chunk::<{ u64!(12) }>() {
            if unlikely(rest.is_empty()) {
                break;
            }

            remainder = rest;

            if !BFAST {
                state[0] ^= read_u64(chunk, u64!(0));
                state[1] ^= read_u64(chunk, u64!(1));
                (lo0, hi0) = wmul(state[0], state[1]);
                state[0] = state[0].wrapping_add(lo5 ^ hi0);

                state[1] ^= read_u64(chunk, u64!(2));
                state[2] ^= read_u64(chunk, u64!(3));
                (lo1, hi1) = wmul(state[1], state[2]);
                state[1] = state[1].wrapping_add(lo0 ^ hi1);

                state[2] ^= read_u64(chunk, u64!(4));
                state[3] ^= read_u64(chunk, u64!(5));
                (lo2, hi2) = wmul(state[2], state[3]);
                state[2] = state[2].wrapping_add(lo1 ^ hi2);

                state[3] ^= read_u64(chunk, u64!(6));
                state[4] ^= read_u64(chunk, u64!(7));
                (lo3, hi3) = wmul(state[3], state[4]);
                state[3] = state[3].wrapping_add(lo2 ^ hi3);

                state[4] ^= read_u64(chunk, u64!(8));
                state[5] ^= read_u64(chunk, u64!(9));
                (lo4, hi4) = wmul(state[4], state[5]);
                state[4] = state[4].wrapping_add(lo3 ^ hi4);

                state[5] ^= read_u64(chunk, u64!(10));
                state[0] ^= read_u64(chunk, u64!(11));
                (lo5, hi5) = wmul(state[5], state[0]);
                state[5] = state[5].wrapping_add(lo4 ^ hi5);
            } else {
                state[0] ^= read_u64(chunk, u64!(0));
                state[1] ^= read_u64(chunk, u64!(1));
                (lo0, hi0) = wmul(state[0], state[1]);
                state[0] = lo5 ^ hi0;

                state[1] ^= read_u64(chunk, u64!(2));
                state[2] ^= read_u64(chunk, u64!(3));
                (lo1, hi1) = wmul(state[1], state[2]);
                state[1] = lo0 ^ hi1;

                state[2] ^= read_u64(chunk, u64!(4));
                state[3] ^= read_u64(chunk, u64!(5));
                (lo2, hi2) = wmul(state[2], state[3]);
                state[2] = lo1 ^ hi2;

                state[3] ^= read_u64(chunk, u64!(6));
                state[4] ^= read_u64(chunk, u64!(7));
                (lo3, hi3) = wmul(state[3], state[4]);
                state[3] = lo2 ^ hi3;

                state[4] ^= read_u64(chunk, u64!(8));
                state[5] ^= read_u64(chunk, u64!(9));
                (lo4, hi4) = wmul(state[4], state[5]);
                state[4] = lo3 ^ hi4;

                state[5] ^= read_u64(chunk, u64!(10));
                state[0] ^= read_u64(chunk, u64!(11));
                (lo5, hi5) = wmul(state[5], state[0]);
                state[5] = lo4 ^ hi5;
            }
        }

        state[0] ^= lo5; // don't forget this!
    }

    [lo0, lo1, lo2, lo3] = [0; 4];
    [hi0, hi1, hi2, hi3] = [0; 4];

    if likely(remainder.len() > u64!(4)) {
        state[0] ^= read_u64(remainder, u64!(0));
        state[1] ^= read_u64(remainder, u64!(1));
        (lo0, hi0) = wmul(state[0], state[1]);

        if likely(remainder.len() > u64!(6)) {
            state[1] ^= read_u64(remainder, u64!(2));
            state[2] ^= read_u64(remainder, u64!(3));
            (lo1, hi1) = wmul(state[1], state[2]);

            if likely(remainder.len() > u64!(8)) {
                state[2] ^= read_u64(remainder, u64!(4));
                state[3] ^= read_u64(remainder, u64!(5));
                (lo2, hi2) = wmul(state[2], state[3]);

                if likely(remainder.len() > u64!(10)) {
                    state[3] ^= read_u64(remainder, u64!(6));
                    state[4] ^= read_u64(remainder, u64!(7));
                    (lo3, hi3) = wmul(state[3], state[4]);
                }
            }
        }
    }

    state[4] ^= read_u64_r(bytes, u64!(3));
    state[5] ^= read_u64_r(bytes, u64!(2));
    (lo4, hi4) = wmul(state[4], state[5]);

    state[5] ^= read_u64_r(bytes, u64!(1));
    state[0] ^= read_u64_r(bytes, u64!(0));
    (lo5, hi5) = wmul(state[5], state[0]);

    let mut i = state[0].wrapping_sub(state[1]);
    let mut j = state[2].wrapping_sub(state[3]);
    let mut k = state[4].wrapping_sub(state[5]);

    let rot = bytes.len() as u32 & 63;
    i = i.rotate_left(rot);
    j = j.rotate_right(rot);
    k ^= bytes.len() as u64;

    i = i.wrapping_add(lo3 ^ hi3 ^ lo4 ^ hi4);
    j = j.wrapping_add(lo5 ^ hi5 ^ lo0 ^ hi0);
    k = k.wrapping_add(lo1 ^ hi1 ^ lo2 ^ hi2);

    (i, j, k)
}

//------------------------------------------------------------------------------

/// Implementation details of the MuseAir hashing algorithm.
pub mod impls {
    use super::*;

    /// Current version of the MuseAir hashing algorithm.
    ///
    /// Note that this refers to the algorithm version, not the implementation version.
    ///
    /// For historical versions, see [`museair.cpp`](https://github.com/eternal-io/museair/blob/master/museair.cpp).
    pub const ALGORITHM_VERSION: &str = "0.4-rc4";

    /// Computes the 64-bit MuseAir hash for a byte slice.
    ///
    /// For most use cases, prefer [`hash`] or [`bfast::hash`] instead.
    #[inline(always)]
    pub const fn hash_64<const BFAST: bool>(bytes: &[u8], seed: u64) -> u64 {
        if likely(bytes.len() <= u64!(4)) {
            hash_short_64::<BFAST>(bytes, seed)
        } else {
            hash_loong_64::<BFAST>(bytes, seed)
        }
    }

    /// Computes the 128-bit MuseAir hash for a byte slice.
    ///
    /// For most use cases, prefer [`hash`] or [`bfast::hash`] instead.
    #[inline(always)]
    pub const fn hash_128<const BFAST: bool>(bytes: &[u8], seed: u64) -> u128 {
        if likely(bytes.len() <= u64!(4)) {
            hash_short_128(bytes, seed)
        } else {
            hash_loong_128::<BFAST>(bytes, seed)
        }
    }

    /// A [`BuildHasher`] for the MuseAir [`IncrementalHasher`] that always has the same fixed seed.
    ///
    /// Note that incrementally hashing small keys will be significantly slower than one-shot hashing.
    ///
    /// For better hashmap performance on small keys and enhanced HashDoS resistance,
    /// consider using [`musemap`](https://crates.io/crates/musemap).
    ///
    /// For most use cases, prefer [`crate::FixedState`] or [`bfast::FixedState`] instead.
    ///
    /// [`BuildHasher`]: core::hash::BuildHasher
    #[derive(Debug, Clone, PartialEq, Eq, Default)]
    pub struct FixedState<const BFAST: bool> {
        per_hasher_seed: u64,
    }

    impl<const BFAST: bool> FixedState<BFAST> {
        /// Creates a `FixedState` with the given per-hasher seed.
        pub const fn new(seed: u64) -> Self {
            Self { per_hasher_seed: seed }
        }
    }

    impl<const BFAST: bool> core::hash::BuildHasher for FixedState<BFAST> {
        type Hasher = IncrementalHasher<BFAST>;

        fn build_hasher(&self) -> Self::Hasher {
            Self::Hasher::new(self.per_hasher_seed)
        }
    }

    /// An incremental [`hasher`](core::hash::Hasher) instance that uses the MuseAir hashing algorithm.
    ///
    /// Note that incrementally hashing small keys will be significantly slower than one-shot hashing.
    ///
    /// For better hashmap performance on small keys and enhanced HashDoS resistance,
    /// consider using [`musemap`](https://crates.io/crates/musemap).
    ///
    /// For most use cases, prefer [`Hasher`] or [`bfast::Hasher`] instead.
    #[derive(Clone)]
    pub struct IncrementalHasher<const BFAST: bool> {
        state: State,
        ring_prev: u64,
        buffer: Chunk,
        buffered_len: usize,
        compressed_len: u64,
    }

    impl<const BFAST: bool> IncrementalHasher<BFAST> {
        /// Create a new MuseAir hasher with a custom seed.
        pub const fn new(seed: u64) -> Self {
            Self {
                state: seed_state(seed),
                ring_prev: CONSTANT[6],
                buffer: [0x00; u64!(12)],
                buffered_len: 0,
                compressed_len: 0,
            }
        }

        /// Write a byte slice to the hasher.
        pub const fn write(&mut self, bytes: &[u8]) {
            let vacancy = self.buffer.len() - self.buffered_len;

            if bytes.len() <= vacancy {
                self.write_little(bytes);
            } else {
                let (bytes, remainder) = bytes.split_at(vacancy);
                self.write_little(bytes);
                self.write_many(remainder);
            }
        }

        /// Returns the 64-bit hash value for the bytes written so far.
        pub const fn finish(&self) -> u64 {
            let tot_len = self.total_len();

            if unlikely(tot_len <= u64!(4)) {
                hash_short_64::<BFAST>(self.remainder(), self.restore_seed())
            } else {
                epilogue_64(self.finalize(tot_len))
            }
        }

        /// Returns the 128-bit hash value for the bytes written so far.
        pub const fn finish_128(&self) -> u128 {
            let tot_len = self.total_len();

            if unlikely(tot_len <= u64!(4)) {
                hash_short_128(self.remainder(), self.restore_seed())
            } else {
                epilogue_128(self.finalize(tot_len))
            }
        }

        #[inline(always)]
        const fn total_len(&self) -> u64 {
            self.compressed_len.wrapping_add(self.buffered_len as u64)
        }

        #[inline(always)]
        const fn restore_seed(&self) -> u64 {
            debug_assert!(self.compressed_len == 0);
            self.state[0].wrapping_sub(CONSTANT[0])
        }

        #[inline(always)]
        const fn remainder(&self) -> &[u8] {
            self.buffer.split_at(self.buffered_len).0
        }

        #[inline(always)]
        const fn end_u64x4(&self) -> [u64; 4] {
            let mid = self.buffered_len;
            let mut helper = [0x00; u64!(4)];

            if mid >= u64!(4) {
                Self::copy_nonoverlapping(self.buffer.split_at(mid - u64!(4)).1.split_at(u64!(4)).0, &mut helper);
            } else {
                let (older, newer) = helper.split_at_mut(u64!(4) - mid);
                Self::copy_nonoverlapping(self.buffer.split_at(mid).0, newer);
                Self::copy_nonoverlapping(self.buffer.split_at(u64!(12) - (u64!(4) - mid)).1, older);
            }

            [
                read_u64_r(&helper, u64!(3)),
                read_u64_r(&helper, u64!(2)),
                read_u64_r(&helper, u64!(1)),
                read_u64_r(&helper, u64!(0)),
            ]
        }

        #[inline(always)]
        const fn copy_nonoverlapping(src: &[u8], dst: &mut [u8]) {
            debug_assert!(src.len() == dst.len());
            let mut i = 0;
            while i < src.len() {
                dst[i] = src[i];
                i += 1;
            }
        }

        #[inline(always)]
        const fn write_little(&mut self, bytes: &[u8]) {
            debug_assert!(self.buffered_len + bytes.len() <= self.buffer.len());
            Self::copy_nonoverlapping(
                bytes,
                self.buffer
                    .split_at_mut(self.buffered_len)
                    .1
                    .split_at_mut(bytes.len())
                    .0,
            );
            self.buffered_len += bytes.len();
        }

        #[inline(never)]
        const fn write_many(&mut self, mut remainder: &[u8]) {
            debug_assert!(self.buffered_len == self.buffer.len() && !remainder.is_empty());

            self.buffered_len = 0;
            let mut first = true;
            let mut state = self.state;

            loop {
                let chunk = match first {
                    true => {
                        first = false;
                        &self.buffer
                    }
                    false => match remainder.split_first_chunk::<{ u64!(12) }>() {
                        None => break,
                        Some((chunk, rest)) => {
                            if unlikely(rest.is_empty()) {
                                break;
                            }
                            remainder = rest;
                            chunk
                        }
                    },
                };

                if !BFAST {
                    state[0] ^= read_u64(chunk, u64!(0));
                    state[1] ^= read_u64(chunk, u64!(1));
                    let (lo0, hi0) = wmul(state[0], state[1]);
                    state[0] = state[0].wrapping_add(self.ring_prev ^ hi0);

                    state[1] ^= read_u64(chunk, u64!(2));
                    state[2] ^= read_u64(chunk, u64!(3));
                    let (lo1, hi1) = wmul(state[1], state[2]);
                    state[1] = state[1].wrapping_add(lo0 ^ hi1);

                    state[2] ^= read_u64(chunk, u64!(4));
                    state[3] ^= read_u64(chunk, u64!(5));
                    let (lo2, hi2) = wmul(state[2], state[3]);
                    state[2] = state[2].wrapping_add(lo1 ^ hi2);

                    state[3] ^= read_u64(chunk, u64!(6));
                    state[4] ^= read_u64(chunk, u64!(7));
                    let (lo3, hi3) = wmul(state[3], state[4]);
                    state[3] = state[3].wrapping_add(lo2 ^ hi3);

                    state[4] ^= read_u64(chunk, u64!(8));
                    state[5] ^= read_u64(chunk, u64!(9));
                    let (lo4, hi4) = wmul(state[4], state[5]);
                    state[4] = state[4].wrapping_add(lo3 ^ hi4);

                    state[5] ^= read_u64(chunk, u64!(10));
                    state[0] ^= read_u64(chunk, u64!(11));
                    let (lo5, hi5) = wmul(state[5], state[0]);
                    state[5] = state[5].wrapping_add(lo4 ^ hi5);

                    self.ring_prev = lo5;
                } else {
                    state[0] ^= read_u64(chunk, u64!(0));
                    state[1] ^= read_u64(chunk, u64!(1));
                    let (lo0, hi0) = wmul(state[0], state[1]);
                    state[0] = self.ring_prev ^ hi0;

                    state[1] ^= read_u64(chunk, u64!(2));
                    state[2] ^= read_u64(chunk, u64!(3));
                    let (lo1, hi1) = wmul(state[1], state[2]);
                    state[1] = lo0 ^ hi1;

                    state[2] ^= read_u64(chunk, u64!(4));
                    state[3] ^= read_u64(chunk, u64!(5));
                    let (lo2, hi2) = wmul(state[2], state[3]);
                    state[2] = lo1 ^ hi2;

                    state[3] ^= read_u64(chunk, u64!(6));
                    state[4] ^= read_u64(chunk, u64!(7));
                    let (lo3, hi3) = wmul(state[3], state[4]);
                    state[3] = lo2 ^ hi3;

                    state[4] ^= read_u64(chunk, u64!(8));
                    state[5] ^= read_u64(chunk, u64!(9));
                    let (lo4, hi4) = wmul(state[4], state[5]);
                    state[4] = lo3 ^ hi4;

                    state[5] ^= read_u64(chunk, u64!(10));
                    state[0] ^= read_u64(chunk, u64!(11));
                    let (lo5, hi5) = wmul(state[5], state[0]);
                    state[5] = lo4 ^ hi5;

                    self.ring_prev = lo5;
                }

                self.compressed_len += u64!(12);
            }

            self.state = state;
            self.write_little(remainder);
        }

        #[inline]
        const fn finalize(&self, tot_len: u64) -> (u64, u64, u64) {
            let [mut lo0, mut lo1, mut lo2, mut lo3, lo4, lo5];
            let [mut hi0, mut hi1, mut hi2, mut hi3, hi4, hi5];

            [lo0, lo1, lo2, lo3] = [0; 4];
            [hi0, hi1, hi2, hi3] = [0; 4];

            let mut state = self.state;
            let remainder = self.remainder();
            let end_u64x4 = self.end_u64x4();

            if self.compressed_len > 0 {
                state[0] ^= self.ring_prev;
            }

            if likely(remainder.len() > u64!(4)) {
                state[0] ^= read_u64(remainder, u64!(0));
                state[1] ^= read_u64(remainder, u64!(1));
                (lo0, hi0) = wmul(state[0], state[1]);

                if likely(remainder.len() > u64!(6)) {
                    state[1] ^= read_u64(remainder, u64!(2));
                    state[2] ^= read_u64(remainder, u64!(3));
                    (lo1, hi1) = wmul(state[1], state[2]);

                    if likely(remainder.len() > u64!(8)) {
                        state[2] ^= read_u64(remainder, u64!(4));
                        state[3] ^= read_u64(remainder, u64!(5));
                        (lo2, hi2) = wmul(state[2], state[3]);

                        if likely(remainder.len() > u64!(10)) {
                            state[3] ^= read_u64(remainder, u64!(6));
                            state[4] ^= read_u64(remainder, u64!(7));
                            (lo3, hi3) = wmul(state[3], state[4]);
                        }
                    }
                }
            }

            state[4] ^= end_u64x4[0];
            state[5] ^= end_u64x4[1];
            (lo4, hi4) = wmul(state[4], state[5]);

            state[5] ^= end_u64x4[2];
            state[0] ^= end_u64x4[3];
            (lo5, hi5) = wmul(state[5], state[0]);

            let mut i = state[0].wrapping_sub(state[1]);
            let mut j = state[2].wrapping_sub(state[3]);
            let mut k = state[4].wrapping_sub(state[5]);

            let rot = tot_len as u32 & 63;
            i = i.rotate_left(rot);
            j = j.rotate_right(rot);
            k ^= tot_len;

            i = i.wrapping_add(lo3 ^ hi3 ^ lo4 ^ hi4);
            j = j.wrapping_add(lo5 ^ hi5 ^ lo0 ^ hi0);
            k = k.wrapping_add(lo1 ^ hi1 ^ lo2 ^ hi2);

            (i, j, k)
        }
    }

    impl<const BFAST: bool> Default for IncrementalHasher<BFAST> {
        fn default() -> Self {
            Self::new(0)
        }
    }

    impl<const BFAST: bool> core::hash::Hasher for IncrementalHasher<BFAST> {
        fn finish(&self) -> u64 {
            self.finish()
        }

        fn write(&mut self, bytes: &[u8]) {
            self.write(bytes);
        }
    }

    impl<const BFAST: bool> core::fmt::Debug for IncrementalHasher<BFAST> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            if BFAST {
                f.debug_struct("MuseAirHasher(BFast)")
                    .field("total_bytes_written", &self.total_len())
                    .finish_non_exhaustive()
            } else {
                f.debug_struct("MuseAirHasher")
                    .field("total_bytes_written", &self.total_len())
                    .finish_non_exhaustive()
            }
        }
    }
}
