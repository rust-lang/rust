#![feature(test)]

extern crate test;
use test::{Bencher, black_box};

extern crate compiler_builtins;
use compiler_builtins::mem::{memcmp, memcpy, memmove, memset};

const WORD_SIZE: usize = core::mem::size_of::<usize>();

struct AlignedVec {
    vec: Vec<usize>,
    size: usize,
}

impl AlignedVec {
    fn new(fill: u8, size: usize) -> Self {
        let mut broadcast = fill as usize;
        let mut bits = 8;
        while bits < WORD_SIZE * 8 {
            broadcast |= broadcast << bits;
            bits *= 2;
        }

        let vec = vec![broadcast; (size + WORD_SIZE - 1) & !WORD_SIZE];
        AlignedVec { vec, size }
    }
}

impl core::ops::Deref for AlignedVec {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.vec.as_ptr() as *const u8, self.size) }
    }
}

impl core::ops::DerefMut for AlignedVec {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.vec.as_mut_ptr() as *mut u8, self.size) }
    }
}

fn memcpy_builtin(b: &mut Bencher, n: usize, offset1: usize, offset2: usize) {
    let v1 = AlignedVec::new(1, n + offset1);
    let mut v2 = AlignedVec::new(0, n + offset2);
    b.bytes = n as u64;
    b.iter(|| {
        let src: &[u8] = black_box(&v1[offset1..]);
        let dst: &mut [u8] = black_box(&mut v2[offset2..]);
        dst.copy_from_slice(src);
    })
}

fn memcpy_rust(b: &mut Bencher, n: usize, offset1: usize, offset2: usize) {
    let v1 = AlignedVec::new(1, n + offset1);
    let mut v2 = AlignedVec::new(0, n + offset2);
    b.bytes = n as u64;
    b.iter(|| {
        let src: &[u8] = black_box(&v1[offset1..]);
        let dst: &mut [u8] = black_box(&mut v2[offset2..]);
        unsafe { memcpy(dst.as_mut_ptr(), src.as_ptr(), n) }
    })
}

fn memset_builtin(b: &mut Bencher, n: usize, offset: usize) {
    let mut v1 = AlignedVec::new(0, n + offset);
    b.bytes = n as u64;
    b.iter(|| {
        let dst: &mut [u8] = black_box(&mut v1[offset..]);
        let val: u8 = black_box(27);
        for b in dst {
            *b = val;
        }
    })
}

fn memset_rust(b: &mut Bencher, n: usize, offset: usize) {
    let mut v1 = AlignedVec::new(0, n + offset);
    b.bytes = n as u64;
    b.iter(|| {
        let dst: &mut [u8] = black_box(&mut v1[offset..]);
        let val = black_box(27);
        unsafe { memset(dst.as_mut_ptr(), val, n) }
    })
}

fn memcmp_builtin(b: &mut Bencher, n: usize) {
    let v1 = AlignedVec::new(0, n);
    let mut v2 = AlignedVec::new(0, n);
    v2[n - 1] = 1;
    b.bytes = n as u64;
    b.iter(|| {
        let s1: &[u8] = black_box(&v1);
        let s2: &[u8] = black_box(&v2);
        s1.cmp(s2)
    })
}

fn memcmp_builtin_unaligned(b: &mut Bencher, n: usize) {
    let v1 = AlignedVec::new(0, n);
    let mut v2 = AlignedVec::new(0, n);
    v2[n - 1] = 1;
    b.bytes = n as u64;
    b.iter(|| {
        let s1: &[u8] = black_box(&v1[0..]);
        let s2: &[u8] = black_box(&v2[1..]);
        s1.cmp(s2)
    })
}

fn memcmp_rust(b: &mut Bencher, n: usize) {
    let v1 = AlignedVec::new(0, n);
    let mut v2 = AlignedVec::new(0, n);
    v2[n - 1] = 1;
    b.bytes = n as u64;
    b.iter(|| {
        let s1: &[u8] = black_box(&v1);
        let s2: &[u8] = black_box(&v2);
        unsafe { memcmp(s1.as_ptr(), s2.as_ptr(), n) }
    })
}

fn memcmp_rust_unaligned(b: &mut Bencher, n: usize) {
    let v1 = AlignedVec::new(0, n);
    let mut v2 = AlignedVec::new(0, n);
    v2[n - 1] = 1;
    b.bytes = n as u64;
    b.iter(|| {
        let s1: &[u8] = black_box(&v1[0..]);
        let s2: &[u8] = black_box(&v2[1..]);
        unsafe { memcmp(s1.as_ptr(), s2.as_ptr(), n - 1) }
    })
}

fn memmove_builtin(b: &mut Bencher, n: usize, offset: usize) {
    let mut v = AlignedVec::new(0, n + n / 2 + offset);
    b.bytes = n as u64;
    b.iter(|| {
        let s: &mut [u8] = black_box(&mut v);
        s.copy_within(0..n, n / 2 + offset);
    })
}

fn memmove_rust(b: &mut Bencher, n: usize, offset: usize) {
    let mut v = AlignedVec::new(0, n + n / 2 + offset);
    b.bytes = n as u64;
    b.iter(|| {
        let dst: *mut u8 = black_box(&mut v[n / 2 + offset..]).as_mut_ptr();
        let src: *const u8 = black_box(&v).as_ptr();
        unsafe { memmove(dst, src, n) };
    })
}

#[bench]
fn memcpy_builtin_4096(b: &mut Bencher) {
    memcpy_builtin(b, 4096, 0, 0)
}
#[bench]
fn memcpy_rust_4096(b: &mut Bencher) {
    memcpy_rust(b, 4096, 0, 0)
}
#[bench]
fn memcpy_builtin_1048576(b: &mut Bencher) {
    memcpy_builtin(b, 1048576, 0, 0)
}
#[bench]
fn memcpy_rust_1048576(b: &mut Bencher) {
    memcpy_rust(b, 1048576, 0, 0)
}
#[bench]
fn memcpy_builtin_4096_offset(b: &mut Bencher) {
    memcpy_builtin(b, 4096, 65, 65)
}
#[bench]
fn memcpy_rust_4096_offset(b: &mut Bencher) {
    memcpy_rust(b, 4096, 65, 65)
}
#[bench]
fn memcpy_builtin_1048576_offset(b: &mut Bencher) {
    memcpy_builtin(b, 1048576, 65, 65)
}
#[bench]
fn memcpy_rust_1048576_offset(b: &mut Bencher) {
    memcpy_rust(b, 1048576, 65, 65)
}
#[bench]
fn memcpy_builtin_4096_misalign(b: &mut Bencher) {
    memcpy_builtin(b, 4096, 65, 66)
}
#[bench]
fn memcpy_rust_4096_misalign(b: &mut Bencher) {
    memcpy_rust(b, 4096, 65, 66)
}
#[bench]
fn memcpy_builtin_1048576_misalign(b: &mut Bencher) {
    memcpy_builtin(b, 1048576, 65, 66)
}
#[bench]
fn memcpy_rust_1048576_misalign(b: &mut Bencher) {
    memcpy_rust(b, 1048576, 65, 66)
}

#[bench]
fn memset_builtin_4096(b: &mut Bencher) {
    memset_builtin(b, 4096, 0)
}
#[bench]
fn memset_rust_4096(b: &mut Bencher) {
    memset_rust(b, 4096, 0)
}
#[bench]
fn memset_builtin_1048576(b: &mut Bencher) {
    memset_builtin(b, 1048576, 0)
}
#[bench]
fn memset_rust_1048576(b: &mut Bencher) {
    memset_rust(b, 1048576, 0)
}
#[bench]
fn memset_builtin_4096_offset(b: &mut Bencher) {
    memset_builtin(b, 4096, 65)
}
#[bench]
fn memset_rust_4096_offset(b: &mut Bencher) {
    memset_rust(b, 4096, 65)
}
#[bench]
fn memset_builtin_1048576_offset(b: &mut Bencher) {
    memset_builtin(b, 1048576, 65)
}
#[bench]
fn memset_rust_1048576_offset(b: &mut Bencher) {
    memset_rust(b, 1048576, 65)
}

#[bench]
fn memcmp_builtin_8(b: &mut Bencher) {
    memcmp_builtin(b, 8)
}
#[bench]
fn memcmp_rust_8(b: &mut Bencher) {
    memcmp_rust(b, 8)
}
#[bench]
fn memcmp_builtin_16(b: &mut Bencher) {
    memcmp_builtin(b, 16)
}
#[bench]
fn memcmp_rust_16(b: &mut Bencher) {
    memcmp_rust(b, 16)
}
#[bench]
fn memcmp_builtin_32(b: &mut Bencher) {
    memcmp_builtin(b, 32)
}
#[bench]
fn memcmp_rust_32(b: &mut Bencher) {
    memcmp_rust(b, 32)
}
#[bench]
fn memcmp_builtin_64(b: &mut Bencher) {
    memcmp_builtin(b, 64)
}
#[bench]
fn memcmp_rust_64(b: &mut Bencher) {
    memcmp_rust(b, 64)
}
#[bench]
fn memcmp_builtin_4096(b: &mut Bencher) {
    memcmp_builtin(b, 4096)
}
#[bench]
fn memcmp_rust_4096(b: &mut Bencher) {
    memcmp_rust(b, 4096)
}
#[bench]
fn memcmp_builtin_1048576(b: &mut Bencher) {
    memcmp_builtin(b, 1048576)
}
#[bench]
fn memcmp_rust_1048576(b: &mut Bencher) {
    memcmp_rust(b, 1048576)
}
#[bench]
fn memcmp_builtin_unaligned_7(b: &mut Bencher) {
    memcmp_builtin_unaligned(b, 8)
}
#[bench]
fn memcmp_rust_unaligned_7(b: &mut Bencher) {
    memcmp_rust_unaligned(b, 8)
}
#[bench]
fn memcmp_builtin_unaligned_15(b: &mut Bencher) {
    memcmp_builtin_unaligned(b, 16)
}
#[bench]
fn memcmp_rust_unaligned_15(b: &mut Bencher) {
    memcmp_rust_unaligned(b, 16)
}
#[bench]
fn memcmp_builtin_unaligned_31(b: &mut Bencher) {
    memcmp_builtin_unaligned(b, 32)
}
#[bench]
fn memcmp_rust_unaligned_31(b: &mut Bencher) {
    memcmp_rust_unaligned(b, 32)
}
#[bench]
fn memcmp_builtin_unaligned_63(b: &mut Bencher) {
    memcmp_builtin_unaligned(b, 64)
}
#[bench]
fn memcmp_rust_unaligned_63(b: &mut Bencher) {
    memcmp_rust_unaligned(b, 64)
}
#[bench]
fn memcmp_builtin_unaligned_4095(b: &mut Bencher) {
    memcmp_builtin_unaligned(b, 4096)
}
#[bench]
fn memcmp_rust_unaligned_4095(b: &mut Bencher) {
    memcmp_rust_unaligned(b, 4096)
}
#[bench]
fn memcmp_builtin_unaligned_1048575(b: &mut Bencher) {
    memcmp_builtin_unaligned(b, 1048576)
}
#[bench]
fn memcmp_rust_unaligned_1048575(b: &mut Bencher) {
    memcmp_rust_unaligned(b, 1048576)
}

#[bench]
fn memmove_builtin_4096(b: &mut Bencher) {
    memmove_builtin(b, 4096, 0)
}
#[bench]
fn memmove_rust_4096(b: &mut Bencher) {
    memmove_rust(b, 4096, 0)
}
#[bench]
fn memmove_builtin_1048576(b: &mut Bencher) {
    memmove_builtin(b, 1048576, 0)
}
#[bench]
fn memmove_rust_1048576(b: &mut Bencher) {
    memmove_rust(b, 1048576, 0)
}
#[bench]
fn memmove_builtin_4096_misalign(b: &mut Bencher) {
    memmove_builtin(b, 4096, 1)
}
#[bench]
fn memmove_rust_4096_misalign(b: &mut Bencher) {
    memmove_rust(b, 4096, 1)
}
#[bench]
fn memmove_builtin_1048576_misalign(b: &mut Bencher) {
    memmove_builtin(b, 1048576, 1)
}
#[bench]
fn memmove_rust_1048576_misalign(b: &mut Bencher) {
    memmove_rust(b, 1048576, 1)
}
