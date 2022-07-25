use super::*;

use std::hash::{Hash, Hasher};

use super::test_data::SINGLE_SHOT_U128;

// Hash just the bytes of the slice, without length prefix
struct Bytes<'a>(&'a [u8]);

impl<'a> Hash for Bytes<'a> {
    #[allow(unused_must_use)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        for byte in self.0 {
            state.write_u8(*byte);
        }
    }
}

fn hash_with<T: Hash>(mut st: Xxh3Hasher, x: &T) -> (u64, u64) {
    x.hash(&mut st);
    st.finish128()
}

fn hash<T: Hash>(x: &T) -> (u64, u64) {
    hash_with(Xxh3Hasher::default(), x)
}

#[test]
#[cfg(target_arch = "arm")]
fn test_hash_usize() {
    let val = 0xdeadbeef_deadbeef_u64;
    assert_ne!(hash(&(val as u64)), hash(&(val as usize)));
    assert_eq!(hash(&(val as u32)), hash(&(val as usize)));
}
#[test]
#[cfg(target_arch = "x86_64")]
fn test_hash_usize() {
    let val = 0xdeadbeef_deadbeef_u64;
    assert_eq!(hash(&(val as u64)), hash(&(val as usize)));
    assert_ne!(hash(&(val as u32)), hash(&(val as usize)));
}
#[test]
#[cfg(target_arch = "x86")]
fn test_hash_usize() {
    let val = 0xdeadbeef_deadbeef_u64;
    assert_ne!(hash(&(val as u64)), hash(&(val as usize)));
    assert_eq!(hash(&(val as u32)), hash(&(val as usize)));
}

#[test]
fn test_hash_idempotent() {
    let val64 = 0xdeadbeef_deadbeef_u64;
    assert_eq!(hash(&val64), hash(&val64));
    let val32 = 0xdeadbeef_u32;
    assert_eq!(hash(&val32), hash(&val32));
}

#[test]
fn test_hash_no_bytes_dropped_64() {
    let val = 0xdeadbeef_deadbeef_u64;

    assert_ne!(hash(&val), hash(&zero_byte(val, 0)));
    assert_ne!(hash(&val), hash(&zero_byte(val, 1)));
    assert_ne!(hash(&val), hash(&zero_byte(val, 2)));
    assert_ne!(hash(&val), hash(&zero_byte(val, 3)));
    assert_ne!(hash(&val), hash(&zero_byte(val, 4)));
    assert_ne!(hash(&val), hash(&zero_byte(val, 5)));
    assert_ne!(hash(&val), hash(&zero_byte(val, 6)));
    assert_ne!(hash(&val), hash(&zero_byte(val, 7)));

    fn zero_byte(val: u64, byte: usize) -> u64 {
        assert!(byte < 8);
        val & !(0xff << (byte * 8))
    }
}

#[test]
fn test_hash_no_bytes_dropped_32() {
    let val = 0xdeadbeef_u32;

    assert_ne!(hash(&val), hash(&zero_byte(val, 0)));
    assert_ne!(hash(&val), hash(&zero_byte(val, 1)));
    assert_ne!(hash(&val), hash(&zero_byte(val, 2)));
    assert_ne!(hash(&val), hash(&zero_byte(val, 3)));

    fn zero_byte(val: u32, byte: usize) -> u32 {
        assert!(byte < 4);
        val & !(0xff << (byte * 8))
    }
}

#[test]
fn test_hash_no_concat_alias() {
    let s = ("aa", "bb");
    let t = ("aabb", "");
    let u = ("a", "abb");

    assert!(s != t && t != u);
    assert!(hash(&s) != hash(&t) && hash(&s) != hash(&u));

    let u = [1, 0, 0, 0];
    let v = (&u[..1], &u[1..3], &u[3..]);
    let w = (&u[..], &u[4..4], &u[4..4]);

    assert_ne!(v, w);
    assert_ne!(hash(&v), hash(&w));
}

#[test]
fn test_short_write_works() {
    let test_u8 = 0xFF_u8;
    let test_u16 = 0x1122_u16;
    let test_u32 = 0x22334455_u32;
    let test_u64 = 0x33445566_778899AA_u64;
    let test_u128 = 0x11223344_55667788_99AABBCC_DDEEFF77_u128;
    let test_usize = 0xD0C0B0A0_usize;

    let test_i8 = -1_i8;
    let test_i16 = -2_i16;
    let test_i32 = -3_i32;
    let test_i64 = -4_i64;
    let test_i128 = -5_i128;
    let test_isize = -6_isize;

    let mut h1 = Xxh3Hasher::default();
    h1.write(b"bytes");
    h1.write(b"string");
    h1.write_u8(test_u8);
    h1.write_u16(test_u16);
    h1.write_u32(test_u32);
    h1.write_u64(test_u64);
    h1.write_u128(test_u128);
    h1.write_usize(test_usize);
    h1.write_i8(test_i8);
    h1.write_i16(test_i16);
    h1.write_i32(test_i32);
    h1.write_i64(test_i64);
    h1.write_i128(test_i128);
    h1.write_isize(test_isize);

    let mut h2 = Xxh3Hasher::default();
    h2.write(b"bytes");
    h2.write(b"string");
    h2.write(&test_u8.to_ne_bytes());
    h2.write(&test_u16.to_ne_bytes());
    h2.write(&test_u32.to_ne_bytes());
    h2.write(&test_u64.to_ne_bytes());
    h2.write(&test_u128.to_ne_bytes());
    h2.write(&test_usize.to_ne_bytes());
    h2.write(&test_i8.to_ne_bytes());
    h2.write(&test_i16.to_ne_bytes());
    h2.write(&test_i32.to_ne_bytes());
    h2.write(&test_i64.to_ne_bytes());
    h2.write(&test_i128.to_ne_bytes());
    h2.write(&test_isize.to_ne_bytes());

    let h1_hash = h1.finish128();
    let h2_hash = h2.finish128();

    assert_eq!(h1_hash, h2_hash);
}

fn test_128(
    bytes: &[u8],
    seed: u64,
    expected_hash: Hash128,
    hash_fn: &dyn Fn(&[u8], u64) -> Hash128,
) {
    let actual_hash = hash_fn(bytes, seed);

    assert!(
        actual_hash == expected_hash,
        "Hash mismatch for input with length {} and seed {}",
        bytes.len(),
        seed
    );
}

fn for_each_test_config(f: &dyn Fn(&[u8], u64, Hash128)) {
    for &(bytes, hashes) in SINGLE_SHOT_U128 {
        for (seed, low64, high64) in hashes {
            if seed != 0 {
                continue;
            }

            f(bytes, seed, Hash128 { low64, high64 });
        }
    }
}

fn test_streaming(accesses: &[usize], seeds: &[u64]) {
    for_each_test_config(&|bytes, seed, expected_hash| {
        if !seeds.contains(&seed) {
            return;
        }

        test_128(bytes, seed, expected_hash, &|mut bytes, _seed| {
            let mut state = Xxh3Hasher::default();

            let mut access_index = 0;

            while bytes.len() > 0 {
                let chunk_len = std::cmp::min(accesses[access_index], bytes.len());

                state.update(&bytes[..chunk_len]);

                bytes = &bytes[chunk_len..];
                access_index = (access_index + 1) % accesses.len();
            }

            let hash = state.digest128();

            hash
        });
    });
}

#[test]
fn single_shot_128() {
    for_each_test_config(&|bytes, seed, hash| {
        if seed == 0 {
            test_128(bytes, seed, hash, &|bytes, _| xxh3_128bits(bytes));
        }
    });
}

#[test]
fn streaming_128() {
    test_streaming(&[1, 2, 3, 4], &[0]);
    test_streaming(&[4, 8, 1, 2, 16, 17], &[0]);
    test_streaming(&[usize::MAX], &[0]);
    test_streaming(&[1], &[0]);
}

pub fn xxh3_128bits(input: &[u8]) -> Hash128 {
    xxh3_128bits_internal(input, 0, &XXH3_K_SECRET.0[..], xxh3_hash_long_128b_default)
}

fn xxh3_hash_long_128b_default(input: &[u8], _seed: u64, _secret: &[u8]) -> Hash128 {
    return xxh3_hash_long_128b_internal(
        input,
        &XXH3_K_SECRET.0[..],
        XXH3_ACCUMULATE_512,
        XXH3_SCRAMBLE_ACC,
    );
}
