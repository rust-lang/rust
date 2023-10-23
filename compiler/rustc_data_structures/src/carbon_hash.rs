use std::hash::{Hash, Hasher};

#[cfg(test)]
mod tests;

pub fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut buf = CarbonHasher::new();
    value.hash(&mut buf);
    buf.hash_code()
}
pub fn hash_value_with_seed<T: Hash>(value: &T, seed: u64) -> u64 {
    let mut buf = CarbonHasher::new_with_seed(seed);
    value.hash(&mut buf);
    buf.hash_code()
}

#[derive(Default)]
pub struct CarbonHasher {
    buffer: u64,
}

const MUL_CONSTANT: u64 = 0x9e37_79b9_7f4a_7c15;

impl CarbonHasher {
    pub fn new_with_seed(seed: u64) -> Self {
        Self { buffer: seed }
    }

    pub fn new() -> Self {
        Self { buffer: 0 }
    }

    pub fn hash_code(self) -> u64 {
        self.buffer
    }

    pub fn hash_sized_bytes(&mut self, bytes: &[u8]) {
        let data_ptr = bytes.as_ptr();
        let size = bytes.len();

        if size == 0 {
            self.hash_one(0);
            return;
        }
        if size <= 8 {
            let data = if size >= 4 {
                unsafe { read_4_to_8(data_ptr, size) }
            } else {
                unsafe { read_1_to_3(data_ptr, size) }
            };

            self.buffer = mix(data ^ self.buffer, sample_random_data(size));
            return;
        }
        if size <= 16 {
            let data = unsafe { read_8_to_16(data_ptr, size) };
            self.buffer = mix(data.0 ^ sample_random_data(size), data.1 ^ self.buffer);
            return;
        }
        if size <= 32 {
            self.buffer ^= sample_random_data(size);
            let m0 = mix(
                unsafe { read_8(data_ptr) } ^ STATIC_RANDOM_DATA[1],
                unsafe { read_8(data_ptr.add(8)) } ^ self.buffer,
            );
            let tail_16b_ptr = unsafe { data_ptr.add(size - 16) };
            let m1 = mix(
                unsafe { read_8(tail_16b_ptr) } ^ STATIC_RANDOM_DATA[3],
                unsafe { read_8(tail_16b_ptr.add(8)) } ^ self.buffer,
            );

            self.buffer = m0 ^ m1;
            return;
        }

        self.hash_bytes_large(bytes);
    }

    pub fn hash_bytes_large(&mut self, bytes: &[u8]) {
        let mut data_ptr = bytes.as_ptr();
        let size = bytes.len() as isize;
        debug_assert!(size > 32);

        unsafe {
            let mix32 = |data_ptr, buffer, random0, random1| {
                let a = read_8(data_ptr);
                let b = read_8(data_ptr.add(8));
                let c = read_8(data_ptr.add(16));
                let d = read_8(data_ptr.add(24));
                let m0 = mix(a ^ random0, b ^ buffer);
                let m1 = mix(c ^ random1, d ^ buffer);
                m0 ^ m1
            };

            let mut buffer0 = self.buffer ^ STATIC_RANDOM_DATA[0];
            let mut buffer1 = self.buffer ^ STATIC_RANDOM_DATA[2];
            let tail_32b_ptr = data_ptr.offset(size - 32);
            let tail_16b_ptr = data_ptr.offset(size - 16);
            let end_ptr = data_ptr.wrapping_offset(size - 64);

            while data_ptr < end_ptr {
                buffer0 = mix32(data_ptr, buffer0, STATIC_RANDOM_DATA[4], STATIC_RANDOM_DATA[5]);
                buffer1 =
                    mix32(data_ptr.add(32), buffer1, STATIC_RANDOM_DATA[6], STATIC_RANDOM_DATA[7]);
                data_ptr = data_ptr.add(64);
            }

            if data_ptr < tail_32b_ptr {
                buffer0 = mix32(data_ptr, buffer0, STATIC_RANDOM_DATA[4], STATIC_RANDOM_DATA[5]);
                data_ptr = data_ptr.add(32);
            }

            if data_ptr < tail_16b_ptr {
                buffer1 =
                    mix32(tail_32b_ptr, buffer1, STATIC_RANDOM_DATA[6], STATIC_RANDOM_DATA[7]);
            } else {
                buffer1 = mix(
                    read_8(tail_16b_ptr) ^ STATIC_RANDOM_DATA[6],
                    read_8(tail_16b_ptr.add(8)) ^ buffer1,
                );
            }

            self.buffer = buffer0 ^ buffer1;
            self.hash_one(size as usize as u64);
        }
    }

    fn hash_one(&mut self, data: u64) {
        self.buffer = mix(data ^ self.buffer, MUL_CONSTANT);
    }
}

fn mix(lhs: u64, rhs: u64) -> u64 {
    let result: u128 = (lhs as u128) * (rhs as u128);
    ((result & (!0_u64) as u128) as u64) ^ ((result >> 64) as u64)
}

unsafe fn read_1_to_3(data: *const u8, size: usize) -> u64 {
    unsafe {
        let byte0 = data.read() as u64;
        let byte1 = data.add(size - 1).read() as u64;
        let byte2 = data.add(size / 2).read() as u64;
        byte0 | (byte1 << 16) | (byte2 << 8)
    }
}

unsafe fn read_4_to_8(data: *const u8, size: usize) -> u64 {
    unsafe {
        let low = data.cast::<u32>().read_unaligned();
        let high = data.add(size).sub(4).cast::<u32>().read_unaligned();
        (low as u64) | ((high as u64) << 32)
    }
}

unsafe fn read_8_to_16(data: *const u8, size: usize) -> (u64, u64) {
    unsafe {
        let low = data.cast::<u64>().read_unaligned();
        let high = data.add(size).sub(8).cast::<u64>().read_unaligned();
        (low, high)
    }
}

unsafe fn read_8(data: *const u8) -> u64 {
    unsafe { data.cast::<u64>().read_unaligned() }
}

fn sample_random_data(offset: usize) -> u64 {
    assert!((offset + std::mem::size_of::<u64>()) < std::mem::size_of::<[u64; 8]>());
    unsafe {
        STATIC_RANDOM_DATA.as_ptr().cast::<[u8; 64]>().add(offset).cast::<u64>().read_unaligned()
    }
}

const STATIC_RANDOM_DATA: &[u64; 8] = &[
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
    0x082e_fa98_ec4e_6c89,
    0x4528_21e6_38d0_1377,
    0xbe54_66cf_34e9_0c6c,
    0xc0ac_29b7_c97c_50dd,
    0x3f84_d5b5_b547_0917,
];

impl Hasher for CarbonHasher {
    fn finish(&self) -> u64 {
        self.buffer
    }

    fn write(&mut self, bytes: &[u8]) {
        self.hash_sized_bytes(bytes);
    }

    fn write_u8(&mut self, i: u8) {
        self.hash_one(i as _);
    }

    fn write_u16(&mut self, i: u16) {
        self.hash_one(i as _);
    }

    fn write_u32(&mut self, i: u32) {
        self.hash_one(i as _);
    }

    fn write_u64(&mut self, i: u64) {
        self.hash_one(i as _);
    }

    fn write_u128(&mut self, i: u128) {
        self.write(&i.to_ne_bytes());
    }

    fn write_usize(&mut self, i: usize) {
        self.hash_one(i as _);
    }

    fn write_i8(&mut self, i: i8) {
        self.hash_one(i as u8 as _);
    }

    fn write_i16(&mut self, i: i16) {
        self.hash_one(i as u16 as _);
    }

    fn write_i32(&mut self, i: i32) {
        self.hash_one(i as u32 as _);
    }

    fn write_i64(&mut self, i: i64) {
        self.hash_one(i as u64 as _);
    }

    fn write_i128(&mut self, i: i128) {
        self.write(&i.to_ne_bytes());
    }

    fn write_isize(&mut self, i: isize) {
        self.hash_one(i as usize as _);
    }
}
