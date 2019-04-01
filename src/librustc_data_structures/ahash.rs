pub use rustc_hash::{FxHasher, FxHashMap, FxHashSet};
use crate::convert::Convert;
use std::hash::{Hasher};
use std::slice;
use std::mem;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use arrayref::*;

const BUFFER_SIZE: usize = 1024;

pub struct BufferedHasher {
    cursor: usize,
    aes: AHasher,
    buffer: [u8; BUFFER_SIZE],
}

impl ::std::fmt::Debug for BufferedHasher {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        write!(f, "{:?} [{}]", self.aes, self.cursor)
    }
}

#[inline(never)]
#[no_mangle]
#[target_feature(enable = "aes")]
unsafe fn ahash_test(input: &[u8]) -> u64 {
    let mut a = AHasher::new_with_keys(67, 87);
    a.write(input);
    a.finish128().0
}

#[inline(never)]
#[no_mangle]
fn ahash_test_one(b: &mut BufferedHasher, input: u64) {
    b.write_u64(input);
}

static AES_ENABLED: AtomicBool = AtomicBool::new(true);

impl BufferedHasher {
    #[inline(always)]
    pub fn new() -> Self {
        BufferedHasher {
            aes: AHasher::new_with_keys(0, 0),
            cursor: 0,
            buffer: unsafe { std::mem::uninitialized() },
        }
    }

    #[inline(always)]
    unsafe fn flush(&mut self) {
        self.aes.write(self.buffer.get_unchecked(0..self.cursor));
    }

    #[inline(never)]
    #[cold]
    unsafe fn flush_cold(&mut self) {
        if likely!(AES_ENABLED.load(Relaxed)) {
            self.flush()
        } else {
            panic!("no aes");
        }
    }

    #[inline(always)]
    fn short_write_gen<T>(&mut self, x: T) {
        let bytes = unsafe {
            slice::from_raw_parts(&x as *const T as *const u8, mem::size_of::<T>())
        };
        self.short_write(bytes);
    }

    #[inline(always)]
    fn short_write(&mut self, data: &[u8]) {
        let mut cursor = self.cursor;
        let len = data.len();
        if unlikely!(cursor + len > BUFFER_SIZE) {
            unsafe { self.flush_cold() };
            cursor = 0;
        }
        unsafe {
            self.buffer.get_unchecked_mut(cursor..(cursor + len)).copy_from_slice(data);
        }
        self.cursor = cursor + len;
    }

    #[inline(never)]
    #[target_feature(enable = "aes")]
    unsafe fn finish128_aes(mut self) -> (u64, u64) {
        self.flush();
        self.aes.finish128()
    }

    #[inline]
    pub fn finish128(self) -> (u64, u64) {
        if likely!(AES_ENABLED.load(Relaxed)) {
            unsafe {
                self.finish128_aes()
            }
        } else {
            panic!("no aes");
        }
    }
}

impl Hasher for BufferedHasher {
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write(&mut self, data: &[u8]) {
        if likely!(data.len() < BUFFER_SIZE / 10) {
            self.short_write(data);
        } else {
            unsafe {
                self.aes.write(data);
            }
        }
    }

    fn finish(&self) -> u64 {
        panic!("cannot provide valid 64 bit hashes")
    }
}

///Just a simple bit pattern.
const PAD : u128 = 0xF0E1_D2C3_B4A5_9687_7869_5A4B_3C2D_1E0F;

#[derive(Debug, Clone)]
pub struct AHasher {
    buffer: [u64; 2],
}

impl AHasher {
    #[inline]
    pub(crate) fn new_with_keys(key0: u64, key1: u64) -> AHasher {
        AHasher { buffer: [key0, key1] }
    }

    #[inline]
    #[target_feature(enable = "aes")]
    unsafe fn write(&mut self, input: &[u8]) {
        let mut data = input;
        let length = data.len() as u64;
        //This will be scrambled by the first AES round in any branch.
        self.buffer[1] ^= length;
        //A 'binary search' on sizes reduces the number of comparisons.
        if data.len() >= 8 {
            if data.len() > 16 {
                if data.len() > 128 {
                    let mut par_block: u128 = self.buffer.convert();
                    while data.len() > 128 {
                        let (b1, rest) = data.split_at(16);
                        let b1: u128 = (*as_array!(b1, 16)).convert();
                        par_block = aeshash(par_block, b1);
                        data = rest;
                        let (b2, rest) = data.split_at(16);
                        let b2: u128 = (*as_array!(b2, 16)).convert();
                        self.buffer = aeshash(self.buffer.convert(), b2).convert();
                        data = rest;
                    }
                    self.buffer = aeshash(self.buffer.convert(), par_block).convert();
                }
                while data.len() > 32 {
                    //len 33-128
                    let (block, rest) = data.split_at(16);
                    let block: u128 = (*as_array!(block, 16)).convert();
                    self.buffer = aeshash(self.buffer.convert(),block).convert();
                    data = rest;
                }
                //len 17-32
                let block = (*array_ref!(data, 0, 16)).convert();
                self.buffer = aeshash(self.buffer.convert(),block).convert();
                let block = (*array_ref!(data, data.len()-16, 16)).convert();
                self.buffer = aeshash(self.buffer.convert(),block).convert();
            } else {
                //len 8-16
                let block: [u64; 2] = [(*array_ref!(data, 0, 8)).convert(),
                    (*array_ref!(data, data.len()-8, 8)).convert()];
                self.buffer = aeshash(self.buffer.convert(),block.convert()).convert();
            }
        } else {
            if data.len() >= 2 {
                if data.len() >= 4 {
                    //len 4-7
                    let block: [u32; 2] = [(*array_ref!(data, 0, 4)).convert(),
                        (*array_ref!(data, data.len()-4, 4)).convert()];
                    let block: [u64;2] = [block[1] as u64, block[0] as u64];
                    self.buffer = aeshash(self.buffer.convert(),block.convert()).convert()
                } else {
                    //len 2-3
                    let block: [u16; 2] = [(*array_ref!(data, 0, 2)).convert(),
                        (*array_ref!(data, data.len()-2, 2)).convert()];
                    let block: u32 = block.convert();
                    self.buffer = aeshash(self.buffer.convert(), block as u128).convert();
                }
            } else {
                if data.len() > 0 {
                    //len 1
                    self.buffer = aeshash(self.buffer.convert(), data[0] as u128).convert();
                }
            }
        }
    }
    #[inline]
    #[target_feature(enable = "aes")]
    unsafe fn finish128(self) -> (u64, u64) {
        let result: [u64; 2] = aeshash(aeshash(self.buffer.convert(), PAD), PAD).convert();
        (result[0], result[1])
    }
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn aeshash(value: u128, xor: u128) -> u128 {
    use std::mem::transmute;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let value = transmute(value);
    transmute(_mm_aesdec_si128(value, transmute(xor)))
}
