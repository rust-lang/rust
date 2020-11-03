#![feature(test)]

extern crate test;
use test::{black_box, Bencher};

extern crate compiler_builtins;
use compiler_builtins::mem::{memcmp, memcpy, memmove, memset};

fn memcpy_builtin(b: &mut Bencher, n: usize, offset: usize) {
    let v1 = vec![1u8; n + offset];
    let mut v2 = vec![0u8; n + offset];
    b.bytes = n as u64;
    b.iter(|| {
        let src: &[u8] = black_box(&v1[offset..]);
        let dst: &mut [u8] = black_box(&mut v2[offset..]);
        dst.copy_from_slice(src);
    })
}

fn memcpy_rust(b: &mut Bencher, n: usize, offset: usize) {
    let v1 = vec![1u8; n + offset];
    let mut v2 = vec![0u8; n + offset];
    b.bytes = n as u64;
    b.iter(|| {
        let src: &[u8] = black_box(&v1[offset..]);
        let dst: &mut [u8] = black_box(&mut v2[offset..]);
        unsafe { memcpy(dst.as_mut_ptr(), src.as_ptr(), n) }
    })
}

fn memset_builtin(b: &mut Bencher, n: usize, offset: usize) {
    let mut v1 = vec![0u8; n + offset];
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
    let mut v1 = vec![0u8; n + offset];
    b.bytes = n as u64;
    b.iter(|| {
        let dst: &mut [u8] = black_box(&mut v1[offset..]);
        let val = black_box(27);
        unsafe { memset(dst.as_mut_ptr(), val, n) }
    })
}

fn memcmp_builtin(b: &mut Bencher, n: usize) {
    let v1 = vec![0u8; n];
    let mut v2 = vec![0u8; n];
    v2[n - 1] = 1;
    b.bytes = n as u64;
    b.iter(|| {
        let s1: &[u8] = black_box(&v1);
        let s2: &[u8] = black_box(&v2);
        s1.cmp(s2)
    })
}

fn memcmp_rust(b: &mut Bencher, n: usize) {
    let v1 = vec![0u8; n];
    let mut v2 = vec![0u8; n];
    v2[n - 1] = 1;
    b.bytes = n as u64;
    b.iter(|| {
        let s1: &[u8] = black_box(&v1);
        let s2: &[u8] = black_box(&v2);
        unsafe { memcmp(s1.as_ptr(), s2.as_ptr(), n) }
    })
}

fn memmove_builtin(b: &mut Bencher, n: usize) {
    let mut v = vec![0u8; n + n / 2];
    b.bytes = n as u64;
    b.iter(|| {
        let s: &mut [u8] = black_box(&mut v);
        s.copy_within(0..n, n / 2);
    })
}

fn memmove_rust(b: &mut Bencher, n: usize) {
    let mut v = vec![0u8; n + n / 2];
    b.bytes = n as u64;
    b.iter(|| {
        let dst: *mut u8 = black_box(&mut v[n / 2..]).as_mut_ptr();
        let src: *const u8 = black_box(&v).as_ptr();
        unsafe { memmove(dst, src, n) };
    })
}

#[bench]
fn memcpy_builtin_4096(b: &mut Bencher) {
    memcpy_builtin(b, 4096, 0)
}
#[bench]
fn memcpy_rust_4096(b: &mut Bencher) {
    memcpy_rust(b, 4096, 0)
}
#[bench]
fn memcpy_builtin_1048576(b: &mut Bencher) {
    memcpy_builtin(b, 1048576, 0)
}
#[bench]
fn memcpy_rust_1048576(b: &mut Bencher) {
    memcpy_rust(b, 1048576, 0)
}
#[bench]
fn memcpy_builtin_4096_offset(b: &mut Bencher) {
    memcpy_builtin(b, 4096, 65)
}
#[bench]
fn memcpy_rust_4096_offset(b: &mut Bencher) {
    memcpy_rust(b, 4096, 65)
}
#[bench]
fn memcpy_builtin_1048576_offset(b: &mut Bencher) {
    memcpy_builtin(b, 1048576, 65)
}
#[bench]
fn memcpy_rust_1048576_offset(b: &mut Bencher) {
    memcpy_rust(b, 1048576, 65)
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
fn memmove_builtin_4096(b: &mut Bencher) {
    memmove_builtin(b, 4096)
}
#[bench]
fn memmove_rust_4096(b: &mut Bencher) {
    memmove_rust(b, 4096)
}
#[bench]
fn memmove_builtin_1048576(b: &mut Bencher) {
    memmove_builtin(b, 1048576)
}
#[bench]
fn memmove_rust_1048576(b: &mut Bencher) {
    memmove_rust(b, 1048576)
}
