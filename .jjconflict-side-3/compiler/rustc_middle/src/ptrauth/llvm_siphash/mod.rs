// LLVM SipHash-2-4 (64 bit version)
pub fn llvm_pointer_auth_stable_siphash(data: &[u8]) -> u16 {
    let raw = llvm_siphash_2_4_64(data);

    ((raw % 0xFFFF) + 1) as u16
}

const SIPHASH_KEY: [u8; 16] = [
    0xb5, 0xd4, 0xc9, 0xeb, 0x79, 0x10, 0x4a, 0x79, 0x6f, 0xec, 0x8b, 0x1b, 0x42, 0x87, 0x81, 0xd4,
];

#[inline(always)]
fn rotl(x: u64, b: u32) -> u64 {
    (x << b) | (x >> (64 - b))
}

#[inline(always)]
fn sipround(v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64) {
    *v0 = v0.wrapping_add(*v1);
    *v1 = rotl(*v1, 13);
    *v1 ^= *v0;
    *v0 = rotl(*v0, 32);

    *v2 = v2.wrapping_add(*v3);
    *v3 = rotl(*v3, 16);
    *v3 ^= *v2;

    *v0 = v0.wrapping_add(*v3);
    *v3 = rotl(*v3, 21);
    *v3 ^= *v0;

    *v2 = v2.wrapping_add(*v1);
    *v1 = rotl(*v1, 17);
    *v1 ^= *v2;
    *v2 = rotl(*v2, 32);
}

fn u64_from_le(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().unwrap())
}

fn load_u64_partial(bytes: &[u8]) -> u64 {
    let mut b = 0u64;

    match bytes.len() {
        7 => {
            b |= (bytes[6] as u64) << 48;
            b |= (bytes[5] as u64) << 40;
            b |= (bytes[4] as u64) << 32;
            b |= (bytes[3] as u64) << 24;
            b |= (bytes[2] as u64) << 16;
            b |= (bytes[1] as u64) << 8;
            b |= bytes[0] as u64;
        }
        6 => {
            b |= (bytes[5] as u64) << 40;
            b |= (bytes[4] as u64) << 32;
            b |= (bytes[3] as u64) << 24;
            b |= (bytes[2] as u64) << 16;
            b |= (bytes[1] as u64) << 8;
            b |= bytes[0] as u64;
        }
        5 => {
            b |= (bytes[4] as u64) << 32;
            b |= (bytes[3] as u64) << 24;
            b |= (bytes[2] as u64) << 16;
            b |= (bytes[1] as u64) << 8;
            b |= bytes[0] as u64;
        }
        4 => {
            b |= (bytes[3] as u64) << 24;
            b |= (bytes[2] as u64) << 16;
            b |= (bytes[1] as u64) << 8;
            b |= bytes[0] as u64;
        }
        3 => {
            b |= (bytes[2] as u64) << 16;
            b |= (bytes[1] as u64) << 8;
            b |= bytes[0] as u64;
        }
        2 => {
            b |= (bytes[1] as u64) << 8;
            b |= bytes[0] as u64;
        }
        1 => {
            b |= bytes[0] as u64;
        }
        _ => {}
    }

    b
}

fn siphash_2_4_64_with_key(data: &[u8], key: &[u8; 16]) -> u64 {
    let k0 = u64_from_le(&key[0..8]);
    let k1 = u64_from_le(&key[8..16]);

    let mut v0: u64 = 0x736f6d6570736575 ^ k0;
    let mut v1: u64 = 0x646f72616e646f6d ^ k1;
    let mut v2: u64 = 0x6c7967656e657261 ^ k0;
    let mut v3: u64 = 0x7465646279746573 ^ k1;

    let mut b: u64 = (data.len() as u64) << 56;
    let mut i = 0;

    // compression
    while i + 8 <= data.len() {
        let m = u64_from_le(&data[i..i + 8]);
        i += 8;

        v3 ^= m;

        for _ in 0..2 {
            sipround(&mut v0, &mut v1, &mut v2, &mut v3);
        }

        v0 ^= m;
    }

    // tail
    let tail = &data[i..];

    b |= load_u64_partial(tail);

    v3 ^= b;

    for _ in 0..2 {
        sipround(&mut v0, &mut v1, &mut v2, &mut v3);
    }

    v0 ^= b;

    // finalization
    v2 ^= 0xff;

    for _ in 0..4 {
        sipround(&mut v0, &mut v1, &mut v2, &mut v3);
    }

    v0 ^ v1 ^ v2 ^ v3
}
// LLVM siphash<2,4> 64-bit output
fn llvm_siphash_2_4_64(data: &[u8]) -> u64 {
    siphash_2_4_64_with_key(data, &SIPHASH_KEY)
}

#[cfg(test)]
mod llvm_siphash_vectors;

#[cfg(test)]
mod tests;
