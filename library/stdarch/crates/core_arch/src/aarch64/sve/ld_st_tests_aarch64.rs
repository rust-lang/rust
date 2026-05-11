// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen-arm/spec/sve` and run the following command to re-generate
// this file:
//
// ```
// cargo run --bin=stdarch-gen-arm -- crates/stdarch-gen-arm/spec
// ```
#![allow(unused)]
use super::*;
use std::boxed::Box;
use std::convert::{TryFrom, TryInto};
use std::sync::LazyLock;
use std::vec::Vec;
use stdarch_test::simd_test;
static F32_DATA: LazyLock<[f32; 64 * 5]> = LazyLock::new(|| {
    (0..64 * 5)
        .map(|i| i as f32)
        .collect::<Vec<_>>()
        .try_into()
        .expect("f32 data incorrectly initialised")
});
static F64_DATA: LazyLock<[f64; 32 * 5]> = LazyLock::new(|| {
    (0..32 * 5)
        .map(|i| i as f64)
        .collect::<Vec<_>>()
        .try_into()
        .expect("f64 data incorrectly initialised")
});
static I8_DATA: LazyLock<[i8; 256 * 5]> = LazyLock::new(|| {
    (0..256 * 5)
        .map(|i| ((i + 128) % 256 - 128) as i8)
        .collect::<Vec<_>>()
        .try_into()
        .expect("i8 data incorrectly initialised")
});
static I16_DATA: LazyLock<[i16; 128 * 5]> = LazyLock::new(|| {
    (0..128 * 5)
        .map(|i| i as i16)
        .collect::<Vec<_>>()
        .try_into()
        .expect("i16 data incorrectly initialised")
});
static I32_DATA: LazyLock<[i32; 64 * 5]> = LazyLock::new(|| {
    (0..64 * 5)
        .map(|i| i as i32)
        .collect::<Vec<_>>()
        .try_into()
        .expect("i32 data incorrectly initialised")
});
static I64_DATA: LazyLock<[i64; 32 * 5]> = LazyLock::new(|| {
    (0..32 * 5)
        .map(|i| i as i64)
        .collect::<Vec<_>>()
        .try_into()
        .expect("i64 data incorrectly initialised")
});
static U8_DATA: LazyLock<[u8; 256 * 5]> = LazyLock::new(|| {
    (0..256 * 5)
        .map(|i| i as u8)
        .collect::<Vec<_>>()
        .try_into()
        .expect("u8 data incorrectly initialised")
});
static U16_DATA: LazyLock<[u16; 128 * 5]> = LazyLock::new(|| {
    (0..128 * 5)
        .map(|i| i as u16)
        .collect::<Vec<_>>()
        .try_into()
        .expect("u16 data incorrectly initialised")
});
static U32_DATA: LazyLock<[u32; 64 * 5]> = LazyLock::new(|| {
    (0..64 * 5)
        .map(|i| i as u32)
        .collect::<Vec<_>>()
        .try_into()
        .expect("u32 data incorrectly initialised")
});
static U64_DATA: LazyLock<[u64; 32 * 5]> = LazyLock::new(|| {
    (0..32 * 5)
        .map(|i| i as u64)
        .collect::<Vec<_>>()
        .try_into()
        .expect("u64 data incorrectly initialised")
});
#[target_feature(enable = "sve")]
fn assert_vector_matches_f32(vector: svfloat32_t, expected: svfloat32_t) {
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b32(), defined));
    let cmp = svcmpne_f32(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}
#[target_feature(enable = "sve")]
fn assert_vector_matches_f64(vector: svfloat64_t, expected: svfloat64_t) {
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b64(), defined));
    let cmp = svcmpne_f64(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}
#[target_feature(enable = "sve")]
fn assert_vector_matches_i8(vector: svint8_t, expected: svint8_t) {
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b8(), defined));
    let cmp = svcmpne_s8(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}
#[target_feature(enable = "sve")]
fn assert_vector_matches_i16(vector: svint16_t, expected: svint16_t) {
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b16(), defined));
    let cmp = svcmpne_s16(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}
#[target_feature(enable = "sve")]
fn assert_vector_matches_i32(vector: svint32_t, expected: svint32_t) {
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b32(), defined));
    let cmp = svcmpne_s32(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}
#[target_feature(enable = "sve")]
fn assert_vector_matches_i64(vector: svint64_t, expected: svint64_t) {
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b64(), defined));
    let cmp = svcmpne_s64(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}
#[target_feature(enable = "sve")]
fn assert_vector_matches_u8(vector: svuint8_t, expected: svuint8_t) {
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b8(), defined));
    let cmp = svcmpne_u8(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}
#[target_feature(enable = "sve")]
fn assert_vector_matches_u16(vector: svuint16_t, expected: svuint16_t) {
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b16(), defined));
    let cmp = svcmpne_u16(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}
#[target_feature(enable = "sve")]
fn assert_vector_matches_u32(vector: svuint32_t, expected: svuint32_t) {
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b32(), defined));
    let cmp = svcmpne_u32(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}
#[target_feature(enable = "sve")]
fn assert_vector_matches_u64(vector: svuint64_t, expected: svuint64_t) {
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b64(), defined));
    let cmp = svcmpne_u64(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_f32_with_svst1_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    svst1_f32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld1_f32(svptrue_b32(), storage.as_ptr() as *const f32);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_f64_with_svst1_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    svst1_f64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld1_f64(svptrue_b64(), storage.as_ptr() as *const f64);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_s8_with_svst1_s8() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s8((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1_s8(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1_s8(svptrue_b8(), storage.as_ptr() as *const i8);
    assert_vector_matches_i8(
        loaded,
        svindex_s8((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_s16_with_svst1_s16() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1_s16(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1_s16(svptrue_b16(), storage.as_ptr() as *const i16);
    assert_vector_matches_i16(
        loaded,
        svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_s32_with_svst1_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1_s32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1_s32(svptrue_b32(), storage.as_ptr() as *const i32);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_s64_with_svst1_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1_s64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld1_s64(svptrue_b64(), storage.as_ptr() as *const i64);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_u8_with_svst1_u8() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u8((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1_u8(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1_u8(svptrue_b8(), storage.as_ptr() as *const u8);
    assert_vector_matches_u8(
        loaded,
        svindex_u8((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_u16_with_svst1_u16() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1_u16(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld1_u16(svptrue_b16(), storage.as_ptr() as *const u16);
    assert_vector_matches_u16(
        loaded,
        svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_u32_with_svst1_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1_u32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld1_u32(svptrue_b32(), storage.as_ptr() as *const u32);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_u64_with_svst1_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1_u64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld1_u64(svptrue_b64(), storage.as_ptr() as *const u64);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_s32index_f32_with_svst1_scatter_s32index_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let indices = svindex_s32(0, 1);
    svst1_scatter_s32index_f32(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld1_gather_s32index_f32(svptrue_b32(), storage.as_ptr() as *const f32, indices);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_s32index_s32_with_svst1_scatter_s32index_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s32(0, 1);
    svst1_scatter_s32index_s32(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1_gather_s32index_s32(svptrue_b32(), storage.as_ptr() as *const i32, indices);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_s32index_u32_with_svst1_scatter_s32index_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s32(0, 1);
    svst1_scatter_s32index_u32(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld1_gather_s32index_u32(svptrue_b32(), storage.as_ptr() as *const u32, indices);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_s64index_f64_with_svst1_scatter_s64index_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let indices = svindex_s64(0, 1);
    svst1_scatter_s64index_f64(svptrue_b64(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld1_gather_s64index_f64(svptrue_b64(), storage.as_ptr() as *const f64, indices);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_s64index_s64_with_svst1_scatter_s64index_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svst1_scatter_s64index_s64(svptrue_b64(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld1_gather_s64index_s64(svptrue_b64(), storage.as_ptr() as *const i64, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_s64index_u64_with_svst1_scatter_s64index_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svst1_scatter_s64index_u64(svptrue_b64(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld1_gather_s64index_u64(svptrue_b64(), storage.as_ptr() as *const u64, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u32index_f32_with_svst1_scatter_u32index_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let indices = svindex_u32(0, 1);
    svst1_scatter_u32index_f32(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld1_gather_u32index_f32(svptrue_b32(), storage.as_ptr() as *const f32, indices);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u32index_s32_with_svst1_scatter_u32index_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u32(0, 1);
    svst1_scatter_u32index_s32(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1_gather_u32index_s32(svptrue_b32(), storage.as_ptr() as *const i32, indices);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u32index_u32_with_svst1_scatter_u32index_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u32(0, 1);
    svst1_scatter_u32index_u32(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld1_gather_u32index_u32(svptrue_b32(), storage.as_ptr() as *const u32, indices);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64index_f64_with_svst1_scatter_u64index_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let indices = svindex_u64(0, 1);
    svst1_scatter_u64index_f64(svptrue_b64(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld1_gather_u64index_f64(svptrue_b64(), storage.as_ptr() as *const f64, indices);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64index_s64_with_svst1_scatter_u64index_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svst1_scatter_u64index_s64(svptrue_b64(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld1_gather_u64index_s64(svptrue_b64(), storage.as_ptr() as *const i64, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64index_u64_with_svst1_scatter_u64index_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svst1_scatter_u64index_u64(svptrue_b64(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld1_gather_u64index_u64(svptrue_b64(), storage.as_ptr() as *const u64, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_s32offset_f32_with_svst1_scatter_s32offset_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let offsets = svindex_s32(0, 4u32.try_into().unwrap());
    svst1_scatter_s32offset_f32(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld1_gather_s32offset_f32(svptrue_b32(), storage.as_ptr() as *const f32, offsets);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_s32offset_s32_with_svst1_scatter_s32offset_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s32(0, 4u32.try_into().unwrap());
    svst1_scatter_s32offset_s32(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1_gather_s32offset_s32(svptrue_b32(), storage.as_ptr() as *const i32, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_s32offset_u32_with_svst1_scatter_s32offset_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s32(0, 4u32.try_into().unwrap());
    svst1_scatter_s32offset_u32(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld1_gather_s32offset_u32(svptrue_b32(), storage.as_ptr() as *const u32, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_s64offset_f64_with_svst1_scatter_s64offset_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let offsets = svindex_s64(0, 8u32.try_into().unwrap());
    svst1_scatter_s64offset_f64(svptrue_b64(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld1_gather_s64offset_f64(svptrue_b64(), storage.as_ptr() as *const f64, offsets);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_s64offset_s64_with_svst1_scatter_s64offset_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 8u32.try_into().unwrap());
    svst1_scatter_s64offset_s64(svptrue_b64(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld1_gather_s64offset_s64(svptrue_b64(), storage.as_ptr() as *const i64, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_s64offset_u64_with_svst1_scatter_s64offset_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 8u32.try_into().unwrap());
    svst1_scatter_s64offset_u64(svptrue_b64(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld1_gather_s64offset_u64(svptrue_b64(), storage.as_ptr() as *const u64, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u32offset_f32_with_svst1_scatter_u32offset_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let offsets = svindex_u32(0, 4u32.try_into().unwrap());
    svst1_scatter_u32offset_f32(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld1_gather_u32offset_f32(svptrue_b32(), storage.as_ptr() as *const f32, offsets);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u32offset_s32_with_svst1_scatter_u32offset_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 4u32.try_into().unwrap());
    svst1_scatter_u32offset_s32(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1_gather_u32offset_s32(svptrue_b32(), storage.as_ptr() as *const i32, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u32offset_u32_with_svst1_scatter_u32offset_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 4u32.try_into().unwrap());
    svst1_scatter_u32offset_u32(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld1_gather_u32offset_u32(svptrue_b32(), storage.as_ptr() as *const u32, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64offset_f64_with_svst1_scatter_u64offset_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    svst1_scatter_u64offset_f64(svptrue_b64(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld1_gather_u64offset_f64(svptrue_b64(), storage.as_ptr() as *const f64, offsets);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64offset_s64_with_svst1_scatter_u64offset_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    svst1_scatter_u64offset_s64(svptrue_b64(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld1_gather_u64offset_s64(svptrue_b64(), storage.as_ptr() as *const i64, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64offset_u64_with_svst1_scatter_u64offset_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    svst1_scatter_u64offset_u64(svptrue_b64(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld1_gather_u64offset_u64(svptrue_b64(), storage.as_ptr() as *const u64, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64base_f64_with_svst1_scatter_u64base_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svst1_scatter_u64base_f64(svptrue_b64(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld1_gather_u64base_f64(svptrue_b64(), bases);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64base_s64_with_svst1_scatter_u64base_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svst1_scatter_u64base_s64(svptrue_b64(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld1_gather_u64base_s64(svptrue_b64(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64base_u64_with_svst1_scatter_u64base_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svst1_scatter_u64base_u64(svptrue_b64(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld1_gather_u64base_u64(svptrue_b64(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u32base_index_f32_with_svst1_scatter_u32base_index_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svst1_scatter_u32base_index_f32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 / (4u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld1_gather_u32base_index_f32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 / (4u32 as i64) + 1,
    );
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u32base_index_s32_with_svst1_scatter_u32base_index_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svst1_scatter_u32base_index_s32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 / (4u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1_gather_u32base_index_s32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 / (4u32 as i64) + 1,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u32base_index_u32_with_svst1_scatter_u32base_index_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svst1_scatter_u32base_index_u32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 / (4u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld1_gather_u32base_index_u32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 / (4u32 as i64) + 1,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64base_index_f64_with_svst1_scatter_u64base_index_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svst1_scatter_u64base_index_f64(svptrue_b64(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld1_gather_u64base_index_f64(svptrue_b64(), bases, 1.try_into().unwrap());
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64base_index_s64_with_svst1_scatter_u64base_index_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svst1_scatter_u64base_index_s64(svptrue_b64(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld1_gather_u64base_index_s64(svptrue_b64(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64base_index_u64_with_svst1_scatter_u64base_index_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svst1_scatter_u64base_index_u64(svptrue_b64(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld1_gather_u64base_index_u64(svptrue_b64(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u32base_offset_f32_with_svst1_scatter_u32base_offset_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svst1_scatter_u32base_offset_f32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 + 4u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld1_gather_u32base_offset_f32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 + 4u32 as i64,
    );
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u32base_offset_s32_with_svst1_scatter_u32base_offset_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svst1_scatter_u32base_offset_s32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 + 4u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1_gather_u32base_offset_s32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 + 4u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u32base_offset_u32_with_svst1_scatter_u32base_offset_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svst1_scatter_u32base_offset_u32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 + 4u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld1_gather_u32base_offset_u32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 + 4u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64base_offset_f64_with_svst1_scatter_u64base_offset_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svst1_scatter_u64base_offset_f64(svptrue_b64(), bases, 8u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld1_gather_u64base_offset_f64(svptrue_b64(), bases, 8u32.try_into().unwrap());
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64base_offset_s64_with_svst1_scatter_u64base_offset_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svst1_scatter_u64base_offset_s64(svptrue_b64(), bases, 8u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld1_gather_u64base_offset_s64(svptrue_b64(), bases, 8u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_gather_u64base_offset_u64_with_svst1_scatter_u64base_offset_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svst1_scatter_u64base_offset_u64(svptrue_b64(), bases, 8u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld1_gather_u64base_offset_u64(svptrue_b64(), bases, 8u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_vnum_f32_with_svst1_vnum_f32() {
    let len = svcntw() as usize;
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
    svst1_vnum_f32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld1_vnum_f32(svptrue_b32(), storage.as_ptr() as *const f32, 1);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 0usize).try_into().unwrap(),
                1usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_vnum_f64_with_svst1_vnum_f64() {
    let len = svcntd() as usize;
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
    svst1_vnum_f64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld1_vnum_f64(svptrue_b64(), storage.as_ptr() as *const f64, 1);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 0usize).try_into().unwrap(),
                1usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_vnum_s8_with_svst1_vnum_s8() {
    let len = svcntb() as usize;
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s8(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1_vnum_s8(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1_vnum_s8(svptrue_b8(), storage.as_ptr() as *const i8, 1);
    assert_vector_matches_i8(
        loaded,
        svindex_s8(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_vnum_s16_with_svst1_vnum_s16() {
    let len = svcnth() as usize;
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s16(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1_vnum_s16(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1_vnum_s16(svptrue_b16(), storage.as_ptr() as *const i16, 1);
    assert_vector_matches_i16(
        loaded,
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_vnum_s32_with_svst1_vnum_s32() {
    let len = svcntw() as usize;
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1_vnum_s32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1_vnum_s32(svptrue_b32(), storage.as_ptr() as *const i32, 1);
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_vnum_s64_with_svst1_vnum_s64() {
    let len = svcntd() as usize;
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1_vnum_s64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld1_vnum_s64(svptrue_b64(), storage.as_ptr() as *const i64, 1);
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_vnum_u8_with_svst1_vnum_u8() {
    let len = svcntb() as usize;
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u8(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1_vnum_u8(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1_vnum_u8(svptrue_b8(), storage.as_ptr() as *const u8, 1);
    assert_vector_matches_u8(
        loaded,
        svindex_u8(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_vnum_u16_with_svst1_vnum_u16() {
    let len = svcnth() as usize;
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u16(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1_vnum_u16(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld1_vnum_u16(svptrue_b16(), storage.as_ptr() as *const u16, 1);
    assert_vector_matches_u16(
        loaded,
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_vnum_u32_with_svst1_vnum_u32() {
    let len = svcntw() as usize;
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1_vnum_u32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld1_vnum_u32(svptrue_b32(), storage.as_ptr() as *const u32, 1);
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1_vnum_u64_with_svst1_vnum_u64() {
    let len = svcntd() as usize;
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1_vnum_u64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld1_vnum_u64(svptrue_b64(), storage.as_ptr() as *const u64, 1);
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve,f64mm")]
unsafe fn test_svld1ro_f32() {
    if svcntb() < 32 {
        println!("Skipping test_svld1ro_f32 due to SVE vector length");
        return;
    }
    svsetffr();
    let loaded = svld1ro_f32(svptrue_b32(), F32_DATA.as_ptr());
    assert_vector_matches_f32(
        loaded,
        svtrn1q_f32(
            svdupq_n_f32(0usize as f32, 1usize as f32, 2usize as f32, 3usize as f32),
            svdupq_n_f32(4usize as f32, 5usize as f32, 6usize as f32, 7usize as f32),
        ),
    );
}
#[simd_test(enable = "sve,f64mm")]
unsafe fn test_svld1ro_f64() {
    if svcntb() < 32 {
        println!("Skipping test_svld1ro_f64 due to SVE vector length");
        return;
    }
    svsetffr();
    let loaded = svld1ro_f64(svptrue_b64(), F64_DATA.as_ptr());
    assert_vector_matches_f64(
        loaded,
        svtrn1q_f64(
            svdupq_n_f64(0usize as f64, 1usize as f64),
            svdupq_n_f64(2usize as f64, 3usize as f64),
        ),
    );
}
#[simd_test(enable = "sve,f64mm")]
unsafe fn test_svld1ro_s8() {
    if svcntb() < 32 {
        println!("Skipping test_svld1ro_s8 due to SVE vector length");
        return;
    }
    svsetffr();
    let loaded = svld1ro_s8(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_i8(
        loaded,
        svtrn1q_s8(
            svdupq_n_s8(
                0usize as i8,
                1usize as i8,
                2usize as i8,
                3usize as i8,
                4usize as i8,
                5usize as i8,
                6usize as i8,
                7usize as i8,
                8usize as i8,
                9usize as i8,
                10usize as i8,
                11usize as i8,
                12usize as i8,
                13usize as i8,
                14usize as i8,
                15usize as i8,
            ),
            svdupq_n_s8(
                16usize as i8,
                17usize as i8,
                18usize as i8,
                19usize as i8,
                20usize as i8,
                21usize as i8,
                22usize as i8,
                23usize as i8,
                24usize as i8,
                25usize as i8,
                26usize as i8,
                27usize as i8,
                28usize as i8,
                29usize as i8,
                30usize as i8,
                31usize as i8,
            ),
        ),
    );
}
#[simd_test(enable = "sve,f64mm")]
unsafe fn test_svld1ro_s16() {
    if svcntb() < 32 {
        println!("Skipping test_svld1ro_s16 due to SVE vector length");
        return;
    }
    svsetffr();
    let loaded = svld1ro_s16(svptrue_b16(), I16_DATA.as_ptr());
    assert_vector_matches_i16(
        loaded,
        svtrn1q_s16(
            svdupq_n_s16(
                0usize as i16,
                1usize as i16,
                2usize as i16,
                3usize as i16,
                4usize as i16,
                5usize as i16,
                6usize as i16,
                7usize as i16,
            ),
            svdupq_n_s16(
                8usize as i16,
                9usize as i16,
                10usize as i16,
                11usize as i16,
                12usize as i16,
                13usize as i16,
                14usize as i16,
                15usize as i16,
            ),
        ),
    );
}
#[simd_test(enable = "sve,f64mm")]
unsafe fn test_svld1ro_s32() {
    if svcntb() < 32 {
        println!("Skipping test_svld1ro_s32 due to SVE vector length");
        return;
    }
    svsetffr();
    let loaded = svld1ro_s32(svptrue_b32(), I32_DATA.as_ptr());
    assert_vector_matches_i32(
        loaded,
        svtrn1q_s32(
            svdupq_n_s32(0usize as i32, 1usize as i32, 2usize as i32, 3usize as i32),
            svdupq_n_s32(4usize as i32, 5usize as i32, 6usize as i32, 7usize as i32),
        ),
    );
}
#[simd_test(enable = "sve,f64mm")]
unsafe fn test_svld1ro_s64() {
    if svcntb() < 32 {
        println!("Skipping test_svld1ro_s64 due to SVE vector length");
        return;
    }
    svsetffr();
    let loaded = svld1ro_s64(svptrue_b64(), I64_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svtrn1q_s64(
            svdupq_n_s64(0usize as i64, 1usize as i64),
            svdupq_n_s64(2usize as i64, 3usize as i64),
        ),
    );
}
#[simd_test(enable = "sve,f64mm")]
unsafe fn test_svld1ro_u8() {
    if svcntb() < 32 {
        println!("Skipping test_svld1ro_u8 due to SVE vector length");
        return;
    }
    svsetffr();
    let loaded = svld1ro_u8(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_u8(
        loaded,
        svtrn1q_u8(
            svdupq_n_u8(
                0usize as u8,
                1usize as u8,
                2usize as u8,
                3usize as u8,
                4usize as u8,
                5usize as u8,
                6usize as u8,
                7usize as u8,
                8usize as u8,
                9usize as u8,
                10usize as u8,
                11usize as u8,
                12usize as u8,
                13usize as u8,
                14usize as u8,
                15usize as u8,
            ),
            svdupq_n_u8(
                16usize as u8,
                17usize as u8,
                18usize as u8,
                19usize as u8,
                20usize as u8,
                21usize as u8,
                22usize as u8,
                23usize as u8,
                24usize as u8,
                25usize as u8,
                26usize as u8,
                27usize as u8,
                28usize as u8,
                29usize as u8,
                30usize as u8,
                31usize as u8,
            ),
        ),
    );
}
#[simd_test(enable = "sve,f64mm")]
unsafe fn test_svld1ro_u16() {
    if svcntb() < 32 {
        println!("Skipping test_svld1ro_u16 due to SVE vector length");
        return;
    }
    svsetffr();
    let loaded = svld1ro_u16(svptrue_b16(), U16_DATA.as_ptr());
    assert_vector_matches_u16(
        loaded,
        svtrn1q_u16(
            svdupq_n_u16(
                0usize as u16,
                1usize as u16,
                2usize as u16,
                3usize as u16,
                4usize as u16,
                5usize as u16,
                6usize as u16,
                7usize as u16,
            ),
            svdupq_n_u16(
                8usize as u16,
                9usize as u16,
                10usize as u16,
                11usize as u16,
                12usize as u16,
                13usize as u16,
                14usize as u16,
                15usize as u16,
            ),
        ),
    );
}
#[simd_test(enable = "sve,f64mm")]
unsafe fn test_svld1ro_u32() {
    if svcntb() < 32 {
        println!("Skipping test_svld1ro_u32 due to SVE vector length");
        return;
    }
    svsetffr();
    let loaded = svld1ro_u32(svptrue_b32(), U32_DATA.as_ptr());
    assert_vector_matches_u32(
        loaded,
        svtrn1q_u32(
            svdupq_n_u32(0usize as u32, 1usize as u32, 2usize as u32, 3usize as u32),
            svdupq_n_u32(4usize as u32, 5usize as u32, 6usize as u32, 7usize as u32),
        ),
    );
}
#[simd_test(enable = "sve,f64mm")]
unsafe fn test_svld1ro_u64() {
    if svcntb() < 32 {
        println!("Skipping test_svld1ro_u64 due to SVE vector length");
        return;
    }
    svsetffr();
    let loaded = svld1ro_u64(svptrue_b64(), U64_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svtrn1q_u64(
            svdupq_n_u64(0usize as u64, 1usize as u64),
            svdupq_n_u64(2usize as u64, 3usize as u64),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1rq_f32() {
    svsetffr();
    let loaded = svld1rq_f32(svptrue_b32(), F32_DATA.as_ptr());
    assert_vector_matches_f32(
        loaded,
        svdupq_n_f32(0usize as f32, 1usize as f32, 2usize as f32, 3usize as f32),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1rq_f64() {
    svsetffr();
    let loaded = svld1rq_f64(svptrue_b64(), F64_DATA.as_ptr());
    assert_vector_matches_f64(loaded, svdupq_n_f64(0usize as f64, 1usize as f64));
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1rq_s8() {
    svsetffr();
    let loaded = svld1rq_s8(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_i8(
        loaded,
        svdupq_n_s8(
            0usize as i8,
            1usize as i8,
            2usize as i8,
            3usize as i8,
            4usize as i8,
            5usize as i8,
            6usize as i8,
            7usize as i8,
            8usize as i8,
            9usize as i8,
            10usize as i8,
            11usize as i8,
            12usize as i8,
            13usize as i8,
            14usize as i8,
            15usize as i8,
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1rq_s16() {
    svsetffr();
    let loaded = svld1rq_s16(svptrue_b16(), I16_DATA.as_ptr());
    assert_vector_matches_i16(
        loaded,
        svdupq_n_s16(
            0usize as i16,
            1usize as i16,
            2usize as i16,
            3usize as i16,
            4usize as i16,
            5usize as i16,
            6usize as i16,
            7usize as i16,
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1rq_s32() {
    svsetffr();
    let loaded = svld1rq_s32(svptrue_b32(), I32_DATA.as_ptr());
    assert_vector_matches_i32(
        loaded,
        svdupq_n_s32(0usize as i32, 1usize as i32, 2usize as i32, 3usize as i32),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1rq_s64() {
    svsetffr();
    let loaded = svld1rq_s64(svptrue_b64(), I64_DATA.as_ptr());
    assert_vector_matches_i64(loaded, svdupq_n_s64(0usize as i64, 1usize as i64));
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1rq_u8() {
    svsetffr();
    let loaded = svld1rq_u8(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_u8(
        loaded,
        svdupq_n_u8(
            0usize as u8,
            1usize as u8,
            2usize as u8,
            3usize as u8,
            4usize as u8,
            5usize as u8,
            6usize as u8,
            7usize as u8,
            8usize as u8,
            9usize as u8,
            10usize as u8,
            11usize as u8,
            12usize as u8,
            13usize as u8,
            14usize as u8,
            15usize as u8,
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1rq_u16() {
    svsetffr();
    let loaded = svld1rq_u16(svptrue_b16(), U16_DATA.as_ptr());
    assert_vector_matches_u16(
        loaded,
        svdupq_n_u16(
            0usize as u16,
            1usize as u16,
            2usize as u16,
            3usize as u16,
            4usize as u16,
            5usize as u16,
            6usize as u16,
            7usize as u16,
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1rq_u32() {
    svsetffr();
    let loaded = svld1rq_u32(svptrue_b32(), U32_DATA.as_ptr());
    assert_vector_matches_u32(
        loaded,
        svdupq_n_u32(0usize as u32, 1usize as u32, 2usize as u32, 3usize as u32),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1rq_u64() {
    svsetffr();
    let loaded = svld1rq_u64(svptrue_b64(), U64_DATA.as_ptr());
    assert_vector_matches_u64(loaded, svdupq_n_u64(0usize as u64, 1usize as u64));
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_s32offset_s32_with_svst1b_scatter_s32offset_s32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s32(0, 1u32.try_into().unwrap());
    svst1b_scatter_s32offset_s32(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_gather_s32offset_s32(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_s32offset_s32_with_svst1h_scatter_s32offset_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s32(0, 2u32.try_into().unwrap());
    svst1h_scatter_s32offset_s32(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_s32offset_s32(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_s32offset_u32_with_svst1b_scatter_s32offset_u32() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s32(0, 1u32.try_into().unwrap());
    svst1b_scatter_s32offset_u32(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1sb_gather_s32offset_u32(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_s32offset_u32_with_svst1h_scatter_s32offset_u32() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s32(0, 2u32.try_into().unwrap());
    svst1h_scatter_s32offset_u32(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_s32offset_u32(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_s64offset_s64_with_svst1b_scatter_s64offset_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 1u32.try_into().unwrap());
    svst1b_scatter_s64offset_s64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_gather_s64offset_s64(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_s64offset_s64_with_svst1h_scatter_s64offset_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 2u32.try_into().unwrap());
    svst1h_scatter_s64offset_s64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_s64offset_s64(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_s64offset_s64_with_svst1w_scatter_s64offset_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 4u32.try_into().unwrap());
    svst1w_scatter_s64offset_s64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svld1sw_gather_s64offset_s64(svptrue_b32(), storage.as_ptr() as *const i32, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_s64offset_u64_with_svst1b_scatter_s64offset_u64() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 1u32.try_into().unwrap());
    svst1b_scatter_s64offset_u64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1sb_gather_s64offset_u64(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_s64offset_u64_with_svst1h_scatter_s64offset_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 2u32.try_into().unwrap());
    svst1h_scatter_s64offset_u64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_s64offset_u64(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_s64offset_u64_with_svst1w_scatter_s64offset_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 4u32.try_into().unwrap());
    svst1w_scatter_s64offset_u64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svld1sw_gather_s64offset_u64(svptrue_b32(), storage.as_ptr() as *const i32, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_u32offset_s32_with_svst1b_scatter_u32offset_s32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 1u32.try_into().unwrap());
    svst1b_scatter_u32offset_s32(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_gather_u32offset_s32(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u32offset_s32_with_svst1h_scatter_u32offset_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 2u32.try_into().unwrap());
    svst1h_scatter_u32offset_s32(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_u32offset_s32(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_u32offset_u32_with_svst1b_scatter_u32offset_u32() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 1u32.try_into().unwrap());
    svst1b_scatter_u32offset_u32(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1sb_gather_u32offset_u32(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u32offset_u32_with_svst1h_scatter_u32offset_u32() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 2u32.try_into().unwrap());
    svst1h_scatter_u32offset_u32(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_u32offset_u32(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_u64offset_s64_with_svst1b_scatter_u64offset_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    svst1b_scatter_u64offset_s64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_gather_u64offset_s64(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u64offset_s64_with_svst1h_scatter_u64offset_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    svst1h_scatter_u64offset_s64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_u64offset_s64(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_u64offset_s64_with_svst1w_scatter_u64offset_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    svst1w_scatter_u64offset_s64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svld1sw_gather_u64offset_s64(svptrue_b32(), storage.as_ptr() as *const i32, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_u64offset_u64_with_svst1b_scatter_u64offset_u64() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    svst1b_scatter_u64offset_u64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1sb_gather_u64offset_u64(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u64offset_u64_with_svst1h_scatter_u64offset_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    svst1h_scatter_u64offset_u64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_u64offset_u64(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_u64offset_u64_with_svst1w_scatter_u64offset_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    svst1w_scatter_u64offset_u64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svld1sw_gather_u64offset_u64(svptrue_b32(), storage.as_ptr() as *const i32, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_u32base_offset_s32_with_svst1b_scatter_u32base_offset_s32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 1u32.try_into().unwrap());
    svst1b_scatter_u32base_offset_s32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_gather_u32base_offset_s32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u32base_offset_s32_with_svst1h_scatter_u32base_offset_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svst1h_scatter_u32base_offset_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_gather_u32base_offset_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_u32base_offset_u32_with_svst1b_scatter_u32base_offset_u32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 1u32.try_into().unwrap());
    svst1b_scatter_u32base_offset_u32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_gather_u32base_offset_u32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u32base_offset_u32_with_svst1h_scatter_u32base_offset_u32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svst1h_scatter_u32base_offset_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_gather_u32base_offset_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_u64base_offset_s64_with_svst1b_scatter_u64base_offset_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svst1b_scatter_u64base_offset_s64(svptrue_b8(), bases, 1u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_gather_u64base_offset_s64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u64base_offset_s64_with_svst1h_scatter_u64base_offset_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svst1h_scatter_u64base_offset_s64(svptrue_b16(), bases, 2u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_gather_u64base_offset_s64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_u64base_offset_s64_with_svst1w_scatter_u64base_offset_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svst1w_scatter_u64base_offset_s64(svptrue_b32(), bases, 4u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1sw_gather_u64base_offset_s64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_u64base_offset_u64_with_svst1b_scatter_u64base_offset_u64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svst1b_scatter_u64base_offset_u64(svptrue_b8(), bases, 1u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_gather_u64base_offset_u64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u64base_offset_u64_with_svst1h_scatter_u64base_offset_u64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svst1h_scatter_u64base_offset_u64(svptrue_b16(), bases, 2u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_gather_u64base_offset_u64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_u64base_offset_u64_with_svst1w_scatter_u64base_offset_u64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svst1w_scatter_u64base_offset_u64(svptrue_b32(), bases, 4u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1sw_gather_u64base_offset_u64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_u64base_s64_with_svst1b_scatter_u64base_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svst1b_scatter_u64base_s64(svptrue_b8(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_gather_u64base_s64(svptrue_b8(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u64base_s64_with_svst1h_scatter_u64base_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svst1h_scatter_u64base_s64(svptrue_b16(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_gather_u64base_s64(svptrue_b16(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_u64base_s64_with_svst1w_scatter_u64base_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svst1w_scatter_u64base_s64(svptrue_b32(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1sw_gather_u64base_s64(svptrue_b32(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_gather_u64base_u64_with_svst1b_scatter_u64base_u64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svst1b_scatter_u64base_u64(svptrue_b8(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_gather_u64base_u64(svptrue_b8(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u64base_u64_with_svst1h_scatter_u64base_u64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svst1h_scatter_u64base_u64(svptrue_b16(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_gather_u64base_u64(svptrue_b16(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_u64base_u64_with_svst1w_scatter_u64base_u64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svst1w_scatter_u64base_u64(svptrue_b32(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1sw_gather_u64base_u64(svptrue_b32(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_s16_with_svst1b_s16() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1b_s16(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_s16(svptrue_b8(), storage.as_ptr() as *const i8);
    assert_vector_matches_i16(
        loaded,
        svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_s32_with_svst1b_s32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1b_s32(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_s32(svptrue_b8(), storage.as_ptr() as *const i8);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_s32_with_svst1h_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1h_s32(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_s32(svptrue_b16(), storage.as_ptr() as *const i16);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_s64_with_svst1b_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1b_s64(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_s64(svptrue_b8(), storage.as_ptr() as *const i8);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_s64_with_svst1h_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1h_s64(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_s64(svptrue_b16(), storage.as_ptr() as *const i16);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_s64_with_svst1w_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1w_s64(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1sw_s64(svptrue_b32(), storage.as_ptr() as *const i32);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_u16_with_svst1b_u16() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1b_u16(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1sb_u16(svptrue_b8(), storage.as_ptr() as *const i8);
    assert_vector_matches_u16(
        loaded,
        svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_u32_with_svst1b_u32() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1b_u32(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1sb_u32(svptrue_b8(), storage.as_ptr() as *const i8);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_u32_with_svst1h_u32() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1h_u32(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld1sh_u32(svptrue_b16(), storage.as_ptr() as *const i16);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_u64_with_svst1b_u64() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1b_u64(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1sb_u64(svptrue_b8(), storage.as_ptr() as *const i8);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_u64_with_svst1h_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1h_u64(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld1sh_u64(svptrue_b16(), storage.as_ptr() as *const i16);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_u64_with_svst1w_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1w_u64(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld1sw_u64(svptrue_b32(), storage.as_ptr() as *const i32);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_vnum_s16_with_svst1b_vnum_s16() {
    let len = svcnth() as usize;
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s16(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1b_vnum_s16(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_vnum_s16(svptrue_b8(), storage.as_ptr() as *const i8, 1);
    assert_vector_matches_i16(
        loaded,
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_vnum_s32_with_svst1b_vnum_s32() {
    let len = svcntw() as usize;
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1b_vnum_s32(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_vnum_s32(svptrue_b8(), storage.as_ptr() as *const i8, 1);
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_vnum_s32_with_svst1h_vnum_s32() {
    let len = svcntw() as usize;
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1h_vnum_s32(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_vnum_s32(svptrue_b16(), storage.as_ptr() as *const i16, 1);
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_vnum_s64_with_svst1b_vnum_s64() {
    let len = svcntd() as usize;
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1b_vnum_s64(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1sb_vnum_s64(svptrue_b8(), storage.as_ptr() as *const i8, 1);
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_vnum_s64_with_svst1h_vnum_s64() {
    let len = svcntd() as usize;
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1h_vnum_s64(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_vnum_s64(svptrue_b16(), storage.as_ptr() as *const i16, 1);
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_vnum_s64_with_svst1w_vnum_s64() {
    let len = svcntd() as usize;
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1w_vnum_s64(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1sw_vnum_s64(svptrue_b32(), storage.as_ptr() as *const i32, 1);
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_vnum_u16_with_svst1b_vnum_u16() {
    let len = svcnth() as usize;
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u16(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1b_vnum_u16(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1sb_vnum_u16(svptrue_b8(), storage.as_ptr() as *const i8, 1);
    assert_vector_matches_u16(
        loaded,
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_vnum_u32_with_svst1b_vnum_u32() {
    let len = svcntw() as usize;
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u32(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1b_vnum_u32(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1sb_vnum_u32(svptrue_b8(), storage.as_ptr() as *const i8, 1);
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_vnum_u32_with_svst1h_vnum_u32() {
    let len = svcntw() as usize;
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1h_vnum_u32(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld1sh_vnum_u32(svptrue_b16(), storage.as_ptr() as *const i16, 1);
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sb_vnum_u64_with_svst1b_vnum_u64() {
    let len = svcntd() as usize;
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1b_vnum_u64(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1sb_vnum_u64(svptrue_b8(), storage.as_ptr() as *const i8, 1);
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_vnum_u64_with_svst1h_vnum_u64() {
    let len = svcntd() as usize;
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1h_vnum_u64(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld1sh_vnum_u64(svptrue_b16(), storage.as_ptr() as *const i16, 1);
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_vnum_u64_with_svst1w_vnum_u64() {
    let len = svcntd() as usize;
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1w_vnum_u64(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld1sw_vnum_u64(svptrue_b32(), storage.as_ptr() as *const i32, 1);
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_s32index_s32_with_svst1h_scatter_s32index_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s32(0, 1);
    svst1h_scatter_s32index_s32(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_s32index_s32(svptrue_b16(), storage.as_ptr() as *const i16, indices);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_s32index_u32_with_svst1h_scatter_s32index_u32() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s32(0, 1);
    svst1h_scatter_s32index_u32(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_s32index_u32(svptrue_b16(), storage.as_ptr() as *const i16, indices);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_s64index_s64_with_svst1h_scatter_s64index_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svst1h_scatter_s64index_s64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_s64index_s64(svptrue_b16(), storage.as_ptr() as *const i16, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_s64index_s64_with_svst1w_scatter_s64index_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svst1w_scatter_s64index_s64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svld1sw_gather_s64index_s64(svptrue_b32(), storage.as_ptr() as *const i32, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_s64index_u64_with_svst1h_scatter_s64index_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svst1h_scatter_s64index_u64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_s64index_u64(svptrue_b16(), storage.as_ptr() as *const i16, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_s64index_u64_with_svst1w_scatter_s64index_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svst1w_scatter_s64index_u64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svld1sw_gather_s64index_u64(svptrue_b32(), storage.as_ptr() as *const i32, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u32index_s32_with_svst1h_scatter_u32index_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u32(0, 1);
    svst1h_scatter_u32index_s32(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_u32index_s32(svptrue_b16(), storage.as_ptr() as *const i16, indices);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u32index_u32_with_svst1h_scatter_u32index_u32() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u32(0, 1);
    svst1h_scatter_u32index_u32(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_u32index_u32(svptrue_b16(), storage.as_ptr() as *const i16, indices);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u64index_s64_with_svst1h_scatter_u64index_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svst1h_scatter_u64index_s64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_u64index_s64(svptrue_b16(), storage.as_ptr() as *const i16, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_u64index_s64_with_svst1w_scatter_u64index_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svst1w_scatter_u64index_s64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svld1sw_gather_u64index_s64(svptrue_b32(), storage.as_ptr() as *const i32, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u64index_u64_with_svst1h_scatter_u64index_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svst1h_scatter_u64index_u64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1sh_gather_u64index_u64(svptrue_b16(), storage.as_ptr() as *const i16, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_u64index_u64_with_svst1w_scatter_u64index_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svst1w_scatter_u64index_u64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svld1sw_gather_u64index_u64(svptrue_b32(), storage.as_ptr() as *const i32, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u32base_index_s32_with_svst1h_scatter_u32base_index_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svst1h_scatter_u32base_index_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_gather_u32base_index_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u32base_index_u32_with_svst1h_scatter_u32base_index_u32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svst1h_scatter_u32base_index_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_gather_u32base_index_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u64base_index_s64_with_svst1h_scatter_u64base_index_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svst1h_scatter_u64base_index_s64(svptrue_b16(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_gather_u64base_index_s64(svptrue_b16(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_u64base_index_s64_with_svst1w_scatter_u64base_index_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svst1w_scatter_u64base_index_s64(svptrue_b32(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1sw_gather_u64base_index_s64(svptrue_b32(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sh_gather_u64base_index_u64_with_svst1h_scatter_u64base_index_u64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svst1h_scatter_u64base_index_u64(svptrue_b16(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1sh_gather_u64base_index_u64(svptrue_b16(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1sw_gather_u64base_index_u64_with_svst1w_scatter_u64base_index_u64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svst1w_scatter_u64base_index_u64(svptrue_b32(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1sw_gather_u64base_index_u64(svptrue_b32(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_s32offset_s32_with_svst1b_scatter_s32offset_s32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s32(0, 1u32.try_into().unwrap());
    svst1b_scatter_s32offset_s32(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_gather_s32offset_s32(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_s32offset_s32_with_svst1h_scatter_s32offset_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s32(0, 2u32.try_into().unwrap());
    svst1h_scatter_s32offset_s32(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_s32offset_s32(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_s32offset_u32_with_svst1b_scatter_s32offset_u32() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s32(0, 1u32.try_into().unwrap());
    svst1b_scatter_s32offset_u32(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1ub_gather_s32offset_u32(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_s32offset_u32_with_svst1h_scatter_s32offset_u32() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s32(0, 2u32.try_into().unwrap());
    svst1h_scatter_s32offset_u32(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_s32offset_u32(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_s64offset_s64_with_svst1b_scatter_s64offset_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 1u32.try_into().unwrap());
    svst1b_scatter_s64offset_s64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_gather_s64offset_s64(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_s64offset_s64_with_svst1h_scatter_s64offset_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 2u32.try_into().unwrap());
    svst1h_scatter_s64offset_s64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_s64offset_s64(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_s64offset_s64_with_svst1w_scatter_s64offset_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 4u32.try_into().unwrap());
    svst1w_scatter_s64offset_s64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svld1uw_gather_s64offset_s64(svptrue_b32(), storage.as_ptr() as *const u32, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_s64offset_u64_with_svst1b_scatter_s64offset_u64() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 1u32.try_into().unwrap());
    svst1b_scatter_s64offset_u64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1ub_gather_s64offset_u64(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_s64offset_u64_with_svst1h_scatter_s64offset_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 2u32.try_into().unwrap());
    svst1h_scatter_s64offset_u64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_s64offset_u64(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_s64offset_u64_with_svst1w_scatter_s64offset_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 4u32.try_into().unwrap());
    svst1w_scatter_s64offset_u64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svld1uw_gather_s64offset_u64(svptrue_b32(), storage.as_ptr() as *const u32, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_u32offset_s32_with_svst1b_scatter_u32offset_s32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 1u32.try_into().unwrap());
    svst1b_scatter_u32offset_s32(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_gather_u32offset_s32(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u32offset_s32_with_svst1h_scatter_u32offset_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 2u32.try_into().unwrap());
    svst1h_scatter_u32offset_s32(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_u32offset_s32(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_u32offset_u32_with_svst1b_scatter_u32offset_u32() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 1u32.try_into().unwrap());
    svst1b_scatter_u32offset_u32(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1ub_gather_u32offset_u32(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u32offset_u32_with_svst1h_scatter_u32offset_u32() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 2u32.try_into().unwrap());
    svst1h_scatter_u32offset_u32(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_u32offset_u32(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_u64offset_s64_with_svst1b_scatter_u64offset_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    svst1b_scatter_u64offset_s64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_gather_u64offset_s64(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u64offset_s64_with_svst1h_scatter_u64offset_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    svst1h_scatter_u64offset_s64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_u64offset_s64(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_u64offset_s64_with_svst1w_scatter_u64offset_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    svst1w_scatter_u64offset_s64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svld1uw_gather_u64offset_s64(svptrue_b32(), storage.as_ptr() as *const u32, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_u64offset_u64_with_svst1b_scatter_u64offset_u64() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    svst1b_scatter_u64offset_u64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1ub_gather_u64offset_u64(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u64offset_u64_with_svst1h_scatter_u64offset_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    svst1h_scatter_u64offset_u64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_u64offset_u64(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_u64offset_u64_with_svst1w_scatter_u64offset_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    svst1w_scatter_u64offset_u64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svld1uw_gather_u64offset_u64(svptrue_b32(), storage.as_ptr() as *const u32, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_u32base_offset_s32_with_svst1b_scatter_u32base_offset_s32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 1u32.try_into().unwrap());
    svst1b_scatter_u32base_offset_s32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_gather_u32base_offset_s32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u32base_offset_s32_with_svst1h_scatter_u32base_offset_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svst1h_scatter_u32base_offset_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_gather_u32base_offset_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_u32base_offset_u32_with_svst1b_scatter_u32base_offset_u32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 1u32.try_into().unwrap());
    svst1b_scatter_u32base_offset_u32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_gather_u32base_offset_u32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u32base_offset_u32_with_svst1h_scatter_u32base_offset_u32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svst1h_scatter_u32base_offset_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_gather_u32base_offset_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_u64base_offset_s64_with_svst1b_scatter_u64base_offset_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svst1b_scatter_u64base_offset_s64(svptrue_b8(), bases, 1u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_gather_u64base_offset_s64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u64base_offset_s64_with_svst1h_scatter_u64base_offset_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svst1h_scatter_u64base_offset_s64(svptrue_b16(), bases, 2u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_gather_u64base_offset_s64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_u64base_offset_s64_with_svst1w_scatter_u64base_offset_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svst1w_scatter_u64base_offset_s64(svptrue_b32(), bases, 4u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1uw_gather_u64base_offset_s64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_u64base_offset_u64_with_svst1b_scatter_u64base_offset_u64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svst1b_scatter_u64base_offset_u64(svptrue_b8(), bases, 1u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_gather_u64base_offset_u64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u64base_offset_u64_with_svst1h_scatter_u64base_offset_u64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svst1h_scatter_u64base_offset_u64(svptrue_b16(), bases, 2u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_gather_u64base_offset_u64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_u64base_offset_u64_with_svst1w_scatter_u64base_offset_u64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svst1w_scatter_u64base_offset_u64(svptrue_b32(), bases, 4u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1uw_gather_u64base_offset_u64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_u64base_s64_with_svst1b_scatter_u64base_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svst1b_scatter_u64base_s64(svptrue_b8(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_gather_u64base_s64(svptrue_b8(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u64base_s64_with_svst1h_scatter_u64base_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svst1h_scatter_u64base_s64(svptrue_b16(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_gather_u64base_s64(svptrue_b16(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_u64base_s64_with_svst1w_scatter_u64base_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svst1w_scatter_u64base_s64(svptrue_b32(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1uw_gather_u64base_s64(svptrue_b32(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_gather_u64base_u64_with_svst1b_scatter_u64base_u64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svst1b_scatter_u64base_u64(svptrue_b8(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_gather_u64base_u64(svptrue_b8(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u64base_u64_with_svst1h_scatter_u64base_u64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svst1h_scatter_u64base_u64(svptrue_b16(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_gather_u64base_u64(svptrue_b16(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_u64base_u64_with_svst1w_scatter_u64base_u64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svst1w_scatter_u64base_u64(svptrue_b32(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1uw_gather_u64base_u64(svptrue_b32(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_s16_with_svst1b_s16() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1b_s16(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_s16(svptrue_b8(), storage.as_ptr() as *const u8);
    assert_vector_matches_i16(
        loaded,
        svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_s32_with_svst1b_s32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1b_s32(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_s32(svptrue_b8(), storage.as_ptr() as *const u8);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_s32_with_svst1h_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1h_s32(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_s32(svptrue_b16(), storage.as_ptr() as *const u16);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_s64_with_svst1b_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1b_s64(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_s64(svptrue_b8(), storage.as_ptr() as *const u8);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_s64_with_svst1h_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1h_s64(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_s64(svptrue_b16(), storage.as_ptr() as *const u16);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_s64_with_svst1w_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1w_s64(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1uw_s64(svptrue_b32(), storage.as_ptr() as *const u32);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_u16_with_svst1b_u16() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1b_u16(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1ub_u16(svptrue_b8(), storage.as_ptr() as *const u8);
    assert_vector_matches_u16(
        loaded,
        svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_u32_with_svst1b_u32() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1b_u32(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1ub_u32(svptrue_b8(), storage.as_ptr() as *const u8);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_u32_with_svst1h_u32() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1h_u32(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld1uh_u32(svptrue_b16(), storage.as_ptr() as *const u16);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_u64_with_svst1b_u64() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1b_u64(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1ub_u64(svptrue_b8(), storage.as_ptr() as *const u8);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_u64_with_svst1h_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1h_u64(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld1uh_u64(svptrue_b16(), storage.as_ptr() as *const u16);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_u64_with_svst1w_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svst1w_u64(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld1uw_u64(svptrue_b32(), storage.as_ptr() as *const u32);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_vnum_s16_with_svst1b_vnum_s16() {
    let len = svcnth() as usize;
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s16(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1b_vnum_s16(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_vnum_s16(svptrue_b8(), storage.as_ptr() as *const u8, 1);
    assert_vector_matches_i16(
        loaded,
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_vnum_s32_with_svst1b_vnum_s32() {
    let len = svcntw() as usize;
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1b_vnum_s32(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_vnum_s32(svptrue_b8(), storage.as_ptr() as *const u8, 1);
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_vnum_s32_with_svst1h_vnum_s32() {
    let len = svcntw() as usize;
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1h_vnum_s32(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_vnum_s32(svptrue_b16(), storage.as_ptr() as *const u16, 1);
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_vnum_s64_with_svst1b_vnum_s64() {
    let len = svcntd() as usize;
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1b_vnum_s64(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld1ub_vnum_s64(svptrue_b8(), storage.as_ptr() as *const u8, 1);
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_vnum_s64_with_svst1h_vnum_s64() {
    let len = svcntd() as usize;
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1h_vnum_s64(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_vnum_s64(svptrue_b16(), storage.as_ptr() as *const u16, 1);
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_vnum_s64_with_svst1w_vnum_s64() {
    let len = svcntd() as usize;
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1w_vnum_s64(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1uw_vnum_s64(svptrue_b32(), storage.as_ptr() as *const u32, 1);
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_vnum_u16_with_svst1b_vnum_u16() {
    let len = svcnth() as usize;
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u16(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1b_vnum_u16(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1ub_vnum_u16(svptrue_b8(), storage.as_ptr() as *const u8, 1);
    assert_vector_matches_u16(
        loaded,
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_vnum_u32_with_svst1b_vnum_u32() {
    let len = svcntw() as usize;
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u32(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1b_vnum_u32(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1ub_vnum_u32(svptrue_b8(), storage.as_ptr() as *const u8, 1);
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_vnum_u32_with_svst1h_vnum_u32() {
    let len = svcntw() as usize;
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1h_vnum_u32(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld1uh_vnum_u32(svptrue_b16(), storage.as_ptr() as *const u16, 1);
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1ub_vnum_u64_with_svst1b_vnum_u64() {
    let len = svcntd() as usize;
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1b_vnum_u64(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld1ub_vnum_u64(svptrue_b8(), storage.as_ptr() as *const u8, 1);
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_vnum_u64_with_svst1h_vnum_u64() {
    let len = svcntd() as usize;
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1h_vnum_u64(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld1uh_vnum_u64(svptrue_b16(), storage.as_ptr() as *const u16, 1);
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_vnum_u64_with_svst1w_vnum_u64() {
    let len = svcntd() as usize;
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svst1w_vnum_u64(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld1uw_vnum_u64(svptrue_b32(), storage.as_ptr() as *const u32, 1);
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_s32index_s32_with_svst1h_scatter_s32index_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s32(0, 1);
    svst1h_scatter_s32index_s32(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_s32index_s32(svptrue_b16(), storage.as_ptr() as *const u16, indices);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_s32index_u32_with_svst1h_scatter_s32index_u32() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s32(0, 1);
    svst1h_scatter_s32index_u32(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_s32index_u32(svptrue_b16(), storage.as_ptr() as *const u16, indices);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_s64index_s64_with_svst1h_scatter_s64index_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svst1h_scatter_s64index_s64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_s64index_s64(svptrue_b16(), storage.as_ptr() as *const u16, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_s64index_s64_with_svst1w_scatter_s64index_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svst1w_scatter_s64index_s64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svld1uw_gather_s64index_s64(svptrue_b32(), storage.as_ptr() as *const u32, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_s64index_u64_with_svst1h_scatter_s64index_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svst1h_scatter_s64index_u64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_s64index_u64(svptrue_b16(), storage.as_ptr() as *const u16, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_s64index_u64_with_svst1w_scatter_s64index_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svst1w_scatter_s64index_u64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svld1uw_gather_s64index_u64(svptrue_b32(), storage.as_ptr() as *const u32, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u32index_s32_with_svst1h_scatter_u32index_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u32(0, 1);
    svst1h_scatter_u32index_s32(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_u32index_s32(svptrue_b16(), storage.as_ptr() as *const u16, indices);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u32index_u32_with_svst1h_scatter_u32index_u32() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u32(0, 1);
    svst1h_scatter_u32index_u32(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_u32index_u32(svptrue_b16(), storage.as_ptr() as *const u16, indices);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u64index_s64_with_svst1h_scatter_u64index_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svst1h_scatter_u64index_s64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_u64index_s64(svptrue_b16(), storage.as_ptr() as *const u16, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_u64index_s64_with_svst1w_scatter_u64index_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svst1w_scatter_u64index_s64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svld1uw_gather_u64index_s64(svptrue_b32(), storage.as_ptr() as *const u32, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u64index_u64_with_svst1h_scatter_u64index_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svst1h_scatter_u64index_u64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svld1uh_gather_u64index_u64(svptrue_b16(), storage.as_ptr() as *const u16, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_u64index_u64_with_svst1w_scatter_u64index_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svst1w_scatter_u64index_u64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svld1uw_gather_u64index_u64(svptrue_b32(), storage.as_ptr() as *const u32, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u32base_index_s32_with_svst1h_scatter_u32base_index_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svst1h_scatter_u32base_index_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_gather_u32base_index_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u32base_index_u32_with_svst1h_scatter_u32base_index_u32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svst1h_scatter_u32base_index_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_gather_u32base_index_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u64base_index_s64_with_svst1h_scatter_u64base_index_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svst1h_scatter_u64base_index_s64(svptrue_b16(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_gather_u64base_index_s64(svptrue_b16(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_u64base_index_s64_with_svst1w_scatter_u64base_index_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svst1w_scatter_u64base_index_s64(svptrue_b32(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1uw_gather_u64base_index_s64(svptrue_b32(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uh_gather_u64base_index_u64_with_svst1h_scatter_u64base_index_u64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svst1h_scatter_u64base_index_u64(svptrue_b16(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld1uh_gather_u64base_index_u64(svptrue_b16(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld1uw_gather_u64base_index_u64_with_svst1w_scatter_u64base_index_u64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svst1w_scatter_u64base_index_u64(svptrue_b32(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld1uw_gather_u64base_index_u64(svptrue_b32(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_f32_with_svst2_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcreate2_f32(
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        ),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        ),
    );
    svst2_f32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld2_f32(svptrue_b32(), storage.as_ptr() as *const f32);
    assert_vector_matches_f32(
        svget2_f32::<{ 0usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        ),
    );
    assert_vector_matches_f32(
        svget2_f32::<{ 1usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_f64_with_svst2_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcreate2_f64(
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        ),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        ),
    );
    svst2_f64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld2_f64(svptrue_b64(), storage.as_ptr() as *const f64);
    assert_vector_matches_f64(
        svget2_f64::<{ 0usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        ),
    );
    assert_vector_matches_f64(
        svget2_f64::<{ 1usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_s8_with_svst2_s8() {
    let mut storage = [0 as i8; 1280usize];
    let data = svcreate2_s8(
        svindex_s8((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        svindex_s8((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    svst2_s8(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld2_s8(svptrue_b8(), storage.as_ptr() as *const i8);
    assert_vector_matches_i8(
        svget2_s8::<{ 0usize as i32 }>(loaded),
        svindex_s8((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    assert_vector_matches_i8(
        svget2_s8::<{ 1usize as i32 }>(loaded),
        svindex_s8((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_s16_with_svst2_s16() {
    let mut storage = [0 as i16; 640usize];
    let data = svcreate2_s16(
        svindex_s16((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        svindex_s16((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    svst2_s16(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld2_s16(svptrue_b16(), storage.as_ptr() as *const i16);
    assert_vector_matches_i16(
        svget2_s16::<{ 0usize as i32 }>(loaded),
        svindex_s16((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    assert_vector_matches_i16(
        svget2_s16::<{ 1usize as i32 }>(loaded),
        svindex_s16((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_s32_with_svst2_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svcreate2_s32(
        svindex_s32((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        svindex_s32((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    svst2_s32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld2_s32(svptrue_b32(), storage.as_ptr() as *const i32);
    assert_vector_matches_i32(
        svget2_s32::<{ 0usize as i32 }>(loaded),
        svindex_s32((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    assert_vector_matches_i32(
        svget2_s32::<{ 1usize as i32 }>(loaded),
        svindex_s32((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_s64_with_svst2_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svcreate2_s64(
        svindex_s64((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        svindex_s64((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    svst2_s64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld2_s64(svptrue_b64(), storage.as_ptr() as *const i64);
    assert_vector_matches_i64(
        svget2_s64::<{ 0usize as i32 }>(loaded),
        svindex_s64((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    assert_vector_matches_i64(
        svget2_s64::<{ 1usize as i32 }>(loaded),
        svindex_s64((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_u8_with_svst2_u8() {
    let mut storage = [0 as u8; 1280usize];
    let data = svcreate2_u8(
        svindex_u8((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        svindex_u8((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    svst2_u8(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld2_u8(svptrue_b8(), storage.as_ptr() as *const u8);
    assert_vector_matches_u8(
        svget2_u8::<{ 0usize as i32 }>(loaded),
        svindex_u8((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    assert_vector_matches_u8(
        svget2_u8::<{ 1usize as i32 }>(loaded),
        svindex_u8((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_u16_with_svst2_u16() {
    let mut storage = [0 as u16; 640usize];
    let data = svcreate2_u16(
        svindex_u16((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        svindex_u16((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    svst2_u16(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld2_u16(svptrue_b16(), storage.as_ptr() as *const u16);
    assert_vector_matches_u16(
        svget2_u16::<{ 0usize as i32 }>(loaded),
        svindex_u16((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    assert_vector_matches_u16(
        svget2_u16::<{ 1usize as i32 }>(loaded),
        svindex_u16((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_u32_with_svst2_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svcreate2_u32(
        svindex_u32((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        svindex_u32((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    svst2_u32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld2_u32(svptrue_b32(), storage.as_ptr() as *const u32);
    assert_vector_matches_u32(
        svget2_u32::<{ 0usize as i32 }>(loaded),
        svindex_u32((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    assert_vector_matches_u32(
        svget2_u32::<{ 1usize as i32 }>(loaded),
        svindex_u32((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_u64_with_svst2_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svcreate2_u64(
        svindex_u64((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
        svindex_u64((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    svst2_u64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld2_u64(svptrue_b64(), storage.as_ptr() as *const u64);
    assert_vector_matches_u64(
        svget2_u64::<{ 0usize as i32 }>(loaded),
        svindex_u64((0usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
    assert_vector_matches_u64(
        svget2_u64::<{ 1usize as i32 }>(loaded),
        svindex_u64((1usize).try_into().unwrap(), 2usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_vnum_f32_with_svst2_vnum_f32() {
    let len = svcntw() as usize;
    let mut storage = [0 as f32; 320usize];
    let data = svcreate2_f32(
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 0usize).try_into().unwrap(),
                2usize.try_into().unwrap(),
            ),
        ),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 1usize).try_into().unwrap(),
                2usize.try_into().unwrap(),
            ),
        ),
    );
    svst2_vnum_f32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld2_vnum_f32(svptrue_b32(), storage.as_ptr() as *const f32, 1);
    assert_vector_matches_f32(
        svget2_f32::<{ 0usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 0usize).try_into().unwrap(),
                2usize.try_into().unwrap(),
            ),
        ),
    );
    assert_vector_matches_f32(
        svget2_f32::<{ 1usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 1usize).try_into().unwrap(),
                2usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_vnum_f64_with_svst2_vnum_f64() {
    let len = svcntd() as usize;
    let mut storage = [0 as f64; 160usize];
    let data = svcreate2_f64(
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 0usize).try_into().unwrap(),
                2usize.try_into().unwrap(),
            ),
        ),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 1usize).try_into().unwrap(),
                2usize.try_into().unwrap(),
            ),
        ),
    );
    svst2_vnum_f64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld2_vnum_f64(svptrue_b64(), storage.as_ptr() as *const f64, 1);
    assert_vector_matches_f64(
        svget2_f64::<{ 0usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 0usize).try_into().unwrap(),
                2usize.try_into().unwrap(),
            ),
        ),
    );
    assert_vector_matches_f64(
        svget2_f64::<{ 1usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 1usize).try_into().unwrap(),
                2usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_vnum_s8_with_svst2_vnum_s8() {
    let len = svcntb() as usize;
    let mut storage = [0 as i8; 1280usize];
    let data = svcreate2_s8(
        svindex_s8(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
        svindex_s8(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    svst2_vnum_s8(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld2_vnum_s8(svptrue_b8(), storage.as_ptr() as *const i8, 1);
    assert_vector_matches_i8(
        svget2_s8::<{ 0usize as i32 }>(loaded),
        svindex_s8(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i8(
        svget2_s8::<{ 1usize as i32 }>(loaded),
        svindex_s8(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_vnum_s16_with_svst2_vnum_s16() {
    let len = svcnth() as usize;
    let mut storage = [0 as i16; 640usize];
    let data = svcreate2_s16(
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
        svindex_s16(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    svst2_vnum_s16(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld2_vnum_s16(svptrue_b16(), storage.as_ptr() as *const i16, 1);
    assert_vector_matches_i16(
        svget2_s16::<{ 0usize as i32 }>(loaded),
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i16(
        svget2_s16::<{ 1usize as i32 }>(loaded),
        svindex_s16(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_vnum_s32_with_svst2_vnum_s32() {
    let len = svcntw() as usize;
    let mut storage = [0 as i32; 320usize];
    let data = svcreate2_s32(
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
        svindex_s32(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    svst2_vnum_s32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld2_vnum_s32(svptrue_b32(), storage.as_ptr() as *const i32, 1);
    assert_vector_matches_i32(
        svget2_s32::<{ 0usize as i32 }>(loaded),
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i32(
        svget2_s32::<{ 1usize as i32 }>(loaded),
        svindex_s32(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_vnum_s64_with_svst2_vnum_s64() {
    let len = svcntd() as usize;
    let mut storage = [0 as i64; 160usize];
    let data = svcreate2_s64(
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
        svindex_s64(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    svst2_vnum_s64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld2_vnum_s64(svptrue_b64(), storage.as_ptr() as *const i64, 1);
    assert_vector_matches_i64(
        svget2_s64::<{ 0usize as i32 }>(loaded),
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i64(
        svget2_s64::<{ 1usize as i32 }>(loaded),
        svindex_s64(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_vnum_u8_with_svst2_vnum_u8() {
    let len = svcntb() as usize;
    let mut storage = [0 as u8; 1280usize];
    let data = svcreate2_u8(
        svindex_u8(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
        svindex_u8(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    svst2_vnum_u8(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld2_vnum_u8(svptrue_b8(), storage.as_ptr() as *const u8, 1);
    assert_vector_matches_u8(
        svget2_u8::<{ 0usize as i32 }>(loaded),
        svindex_u8(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u8(
        svget2_u8::<{ 1usize as i32 }>(loaded),
        svindex_u8(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_vnum_u16_with_svst2_vnum_u16() {
    let len = svcnth() as usize;
    let mut storage = [0 as u16; 640usize];
    let data = svcreate2_u16(
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
        svindex_u16(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    svst2_vnum_u16(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld2_vnum_u16(svptrue_b16(), storage.as_ptr() as *const u16, 1);
    assert_vector_matches_u16(
        svget2_u16::<{ 0usize as i32 }>(loaded),
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u16(
        svget2_u16::<{ 1usize as i32 }>(loaded),
        svindex_u16(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_vnum_u32_with_svst2_vnum_u32() {
    let len = svcntw() as usize;
    let mut storage = [0 as u32; 320usize];
    let data = svcreate2_u32(
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
        svindex_u32(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    svst2_vnum_u32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld2_vnum_u32(svptrue_b32(), storage.as_ptr() as *const u32, 1);
    assert_vector_matches_u32(
        svget2_u32::<{ 0usize as i32 }>(loaded),
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u32(
        svget2_u32::<{ 1usize as i32 }>(loaded),
        svindex_u32(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld2_vnum_u64_with_svst2_vnum_u64() {
    let len = svcntd() as usize;
    let mut storage = [0 as u64; 160usize];
    let data = svcreate2_u64(
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
        svindex_u64(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    svst2_vnum_u64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld2_vnum_u64(svptrue_b64(), storage.as_ptr() as *const u64, 1);
    assert_vector_matches_u64(
        svget2_u64::<{ 0usize as i32 }>(loaded),
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u64(
        svget2_u64::<{ 1usize as i32 }>(loaded),
        svindex_u64(
            (len + 1usize).try_into().unwrap(),
            2usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_f32_with_svst3_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcreate3_f32(
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        ),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        ),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        ),
    );
    svst3_f32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld3_f32(svptrue_b32(), storage.as_ptr() as *const f32);
    assert_vector_matches_f32(
        svget3_f32::<{ 0usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        ),
    );
    assert_vector_matches_f32(
        svget3_f32::<{ 1usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        ),
    );
    assert_vector_matches_f32(
        svget3_f32::<{ 2usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_f64_with_svst3_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcreate3_f64(
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        ),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        ),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        ),
    );
    svst3_f64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld3_f64(svptrue_b64(), storage.as_ptr() as *const f64);
    assert_vector_matches_f64(
        svget3_f64::<{ 0usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        ),
    );
    assert_vector_matches_f64(
        svget3_f64::<{ 1usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        ),
    );
    assert_vector_matches_f64(
        svget3_f64::<{ 2usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_s8_with_svst3_s8() {
    let mut storage = [0 as i8; 1280usize];
    let data = svcreate3_s8(
        svindex_s8((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_s8((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_s8((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    svst3_s8(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld3_s8(svptrue_b8(), storage.as_ptr() as *const i8);
    assert_vector_matches_i8(
        svget3_s8::<{ 0usize as i32 }>(loaded),
        svindex_s8((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_i8(
        svget3_s8::<{ 1usize as i32 }>(loaded),
        svindex_s8((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_i8(
        svget3_s8::<{ 2usize as i32 }>(loaded),
        svindex_s8((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_s16_with_svst3_s16() {
    let mut storage = [0 as i16; 640usize];
    let data = svcreate3_s16(
        svindex_s16((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_s16((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_s16((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    svst3_s16(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld3_s16(svptrue_b16(), storage.as_ptr() as *const i16);
    assert_vector_matches_i16(
        svget3_s16::<{ 0usize as i32 }>(loaded),
        svindex_s16((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_i16(
        svget3_s16::<{ 1usize as i32 }>(loaded),
        svindex_s16((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_i16(
        svget3_s16::<{ 2usize as i32 }>(loaded),
        svindex_s16((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_s32_with_svst3_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svcreate3_s32(
        svindex_s32((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_s32((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_s32((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    svst3_s32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld3_s32(svptrue_b32(), storage.as_ptr() as *const i32);
    assert_vector_matches_i32(
        svget3_s32::<{ 0usize as i32 }>(loaded),
        svindex_s32((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_i32(
        svget3_s32::<{ 1usize as i32 }>(loaded),
        svindex_s32((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_i32(
        svget3_s32::<{ 2usize as i32 }>(loaded),
        svindex_s32((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_s64_with_svst3_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svcreate3_s64(
        svindex_s64((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_s64((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_s64((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    svst3_s64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld3_s64(svptrue_b64(), storage.as_ptr() as *const i64);
    assert_vector_matches_i64(
        svget3_s64::<{ 0usize as i32 }>(loaded),
        svindex_s64((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_i64(
        svget3_s64::<{ 1usize as i32 }>(loaded),
        svindex_s64((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_i64(
        svget3_s64::<{ 2usize as i32 }>(loaded),
        svindex_s64((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_u8_with_svst3_u8() {
    let mut storage = [0 as u8; 1280usize];
    let data = svcreate3_u8(
        svindex_u8((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_u8((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_u8((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    svst3_u8(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld3_u8(svptrue_b8(), storage.as_ptr() as *const u8);
    assert_vector_matches_u8(
        svget3_u8::<{ 0usize as i32 }>(loaded),
        svindex_u8((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_u8(
        svget3_u8::<{ 1usize as i32 }>(loaded),
        svindex_u8((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_u8(
        svget3_u8::<{ 2usize as i32 }>(loaded),
        svindex_u8((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_u16_with_svst3_u16() {
    let mut storage = [0 as u16; 640usize];
    let data = svcreate3_u16(
        svindex_u16((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_u16((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_u16((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    svst3_u16(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld3_u16(svptrue_b16(), storage.as_ptr() as *const u16);
    assert_vector_matches_u16(
        svget3_u16::<{ 0usize as i32 }>(loaded),
        svindex_u16((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_u16(
        svget3_u16::<{ 1usize as i32 }>(loaded),
        svindex_u16((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_u16(
        svget3_u16::<{ 2usize as i32 }>(loaded),
        svindex_u16((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_u32_with_svst3_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svcreate3_u32(
        svindex_u32((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_u32((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_u32((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    svst3_u32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld3_u32(svptrue_b32(), storage.as_ptr() as *const u32);
    assert_vector_matches_u32(
        svget3_u32::<{ 0usize as i32 }>(loaded),
        svindex_u32((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_u32(
        svget3_u32::<{ 1usize as i32 }>(loaded),
        svindex_u32((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_u32(
        svget3_u32::<{ 2usize as i32 }>(loaded),
        svindex_u32((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_u64_with_svst3_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svcreate3_u64(
        svindex_u64((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_u64((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
        svindex_u64((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    svst3_u64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld3_u64(svptrue_b64(), storage.as_ptr() as *const u64);
    assert_vector_matches_u64(
        svget3_u64::<{ 0usize as i32 }>(loaded),
        svindex_u64((0usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_u64(
        svget3_u64::<{ 1usize as i32 }>(loaded),
        svindex_u64((1usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
    assert_vector_matches_u64(
        svget3_u64::<{ 2usize as i32 }>(loaded),
        svindex_u64((2usize).try_into().unwrap(), 3usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_vnum_f32_with_svst3_vnum_f32() {
    let len = svcntw() as usize;
    let mut storage = [0 as f32; 320usize];
    let data = svcreate3_f32(
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 0usize).try_into().unwrap(),
                3usize.try_into().unwrap(),
            ),
        ),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 1usize).try_into().unwrap(),
                3usize.try_into().unwrap(),
            ),
        ),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 2usize).try_into().unwrap(),
                3usize.try_into().unwrap(),
            ),
        ),
    );
    svst3_vnum_f32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld3_vnum_f32(svptrue_b32(), storage.as_ptr() as *const f32, 1);
    assert_vector_matches_f32(
        svget3_f32::<{ 0usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 0usize).try_into().unwrap(),
                3usize.try_into().unwrap(),
            ),
        ),
    );
    assert_vector_matches_f32(
        svget3_f32::<{ 1usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 1usize).try_into().unwrap(),
                3usize.try_into().unwrap(),
            ),
        ),
    );
    assert_vector_matches_f32(
        svget3_f32::<{ 2usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 2usize).try_into().unwrap(),
                3usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_vnum_f64_with_svst3_vnum_f64() {
    let len = svcntd() as usize;
    let mut storage = [0 as f64; 160usize];
    let data = svcreate3_f64(
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 0usize).try_into().unwrap(),
                3usize.try_into().unwrap(),
            ),
        ),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 1usize).try_into().unwrap(),
                3usize.try_into().unwrap(),
            ),
        ),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 2usize).try_into().unwrap(),
                3usize.try_into().unwrap(),
            ),
        ),
    );
    svst3_vnum_f64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld3_vnum_f64(svptrue_b64(), storage.as_ptr() as *const f64, 1);
    assert_vector_matches_f64(
        svget3_f64::<{ 0usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 0usize).try_into().unwrap(),
                3usize.try_into().unwrap(),
            ),
        ),
    );
    assert_vector_matches_f64(
        svget3_f64::<{ 1usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 1usize).try_into().unwrap(),
                3usize.try_into().unwrap(),
            ),
        ),
    );
    assert_vector_matches_f64(
        svget3_f64::<{ 2usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 2usize).try_into().unwrap(),
                3usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_vnum_s8_with_svst3_vnum_s8() {
    let len = svcntb() as usize;
    let mut storage = [0 as i8; 1280usize];
    let data = svcreate3_s8(
        svindex_s8(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_s8(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_s8(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    svst3_vnum_s8(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld3_vnum_s8(svptrue_b8(), storage.as_ptr() as *const i8, 1);
    assert_vector_matches_i8(
        svget3_s8::<{ 0usize as i32 }>(loaded),
        svindex_s8(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i8(
        svget3_s8::<{ 1usize as i32 }>(loaded),
        svindex_s8(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i8(
        svget3_s8::<{ 2usize as i32 }>(loaded),
        svindex_s8(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_vnum_s16_with_svst3_vnum_s16() {
    let len = svcnth() as usize;
    let mut storage = [0 as i16; 640usize];
    let data = svcreate3_s16(
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_s16(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_s16(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    svst3_vnum_s16(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld3_vnum_s16(svptrue_b16(), storage.as_ptr() as *const i16, 1);
    assert_vector_matches_i16(
        svget3_s16::<{ 0usize as i32 }>(loaded),
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i16(
        svget3_s16::<{ 1usize as i32 }>(loaded),
        svindex_s16(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i16(
        svget3_s16::<{ 2usize as i32 }>(loaded),
        svindex_s16(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_vnum_s32_with_svst3_vnum_s32() {
    let len = svcntw() as usize;
    let mut storage = [0 as i32; 320usize];
    let data = svcreate3_s32(
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_s32(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_s32(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    svst3_vnum_s32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld3_vnum_s32(svptrue_b32(), storage.as_ptr() as *const i32, 1);
    assert_vector_matches_i32(
        svget3_s32::<{ 0usize as i32 }>(loaded),
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i32(
        svget3_s32::<{ 1usize as i32 }>(loaded),
        svindex_s32(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i32(
        svget3_s32::<{ 2usize as i32 }>(loaded),
        svindex_s32(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_vnum_s64_with_svst3_vnum_s64() {
    let len = svcntd() as usize;
    let mut storage = [0 as i64; 160usize];
    let data = svcreate3_s64(
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_s64(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_s64(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    svst3_vnum_s64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld3_vnum_s64(svptrue_b64(), storage.as_ptr() as *const i64, 1);
    assert_vector_matches_i64(
        svget3_s64::<{ 0usize as i32 }>(loaded),
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i64(
        svget3_s64::<{ 1usize as i32 }>(loaded),
        svindex_s64(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i64(
        svget3_s64::<{ 2usize as i32 }>(loaded),
        svindex_s64(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_vnum_u8_with_svst3_vnum_u8() {
    let len = svcntb() as usize;
    let mut storage = [0 as u8; 1280usize];
    let data = svcreate3_u8(
        svindex_u8(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_u8(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_u8(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    svst3_vnum_u8(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld3_vnum_u8(svptrue_b8(), storage.as_ptr() as *const u8, 1);
    assert_vector_matches_u8(
        svget3_u8::<{ 0usize as i32 }>(loaded),
        svindex_u8(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u8(
        svget3_u8::<{ 1usize as i32 }>(loaded),
        svindex_u8(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u8(
        svget3_u8::<{ 2usize as i32 }>(loaded),
        svindex_u8(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_vnum_u16_with_svst3_vnum_u16() {
    let len = svcnth() as usize;
    let mut storage = [0 as u16; 640usize];
    let data = svcreate3_u16(
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_u16(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_u16(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    svst3_vnum_u16(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld3_vnum_u16(svptrue_b16(), storage.as_ptr() as *const u16, 1);
    assert_vector_matches_u16(
        svget3_u16::<{ 0usize as i32 }>(loaded),
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u16(
        svget3_u16::<{ 1usize as i32 }>(loaded),
        svindex_u16(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u16(
        svget3_u16::<{ 2usize as i32 }>(loaded),
        svindex_u16(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_vnum_u32_with_svst3_vnum_u32() {
    let len = svcntw() as usize;
    let mut storage = [0 as u32; 320usize];
    let data = svcreate3_u32(
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_u32(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_u32(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    svst3_vnum_u32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld3_vnum_u32(svptrue_b32(), storage.as_ptr() as *const u32, 1);
    assert_vector_matches_u32(
        svget3_u32::<{ 0usize as i32 }>(loaded),
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u32(
        svget3_u32::<{ 1usize as i32 }>(loaded),
        svindex_u32(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u32(
        svget3_u32::<{ 2usize as i32 }>(loaded),
        svindex_u32(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld3_vnum_u64_with_svst3_vnum_u64() {
    let len = svcntd() as usize;
    let mut storage = [0 as u64; 160usize];
    let data = svcreate3_u64(
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_u64(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
        svindex_u64(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    svst3_vnum_u64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld3_vnum_u64(svptrue_b64(), storage.as_ptr() as *const u64, 1);
    assert_vector_matches_u64(
        svget3_u64::<{ 0usize as i32 }>(loaded),
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u64(
        svget3_u64::<{ 1usize as i32 }>(loaded),
        svindex_u64(
            (len + 1usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u64(
        svget3_u64::<{ 2usize as i32 }>(loaded),
        svindex_u64(
            (len + 2usize).try_into().unwrap(),
            3usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_f32_with_svst4_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcreate4_f32(
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
    );
    svst4_f32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld4_f32(svptrue_b32(), storage.as_ptr() as *const f32);
    assert_vector_matches_f32(
        svget4_f32::<{ 0usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
    );
    assert_vector_matches_f32(
        svget4_f32::<{ 1usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
    );
    assert_vector_matches_f32(
        svget4_f32::<{ 2usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
    );
    assert_vector_matches_f32(
        svget4_f32::<{ 3usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_f64_with_svst4_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcreate4_f64(
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
    );
    svst4_f64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld4_f64(svptrue_b64(), storage.as_ptr() as *const f64);
    assert_vector_matches_f64(
        svget4_f64::<{ 0usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
    );
    assert_vector_matches_f64(
        svget4_f64::<{ 1usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
    );
    assert_vector_matches_f64(
        svget4_f64::<{ 2usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
    );
    assert_vector_matches_f64(
        svget4_f64::<{ 3usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_s8_with_svst4_s8() {
    let mut storage = [0 as i8; 1280usize];
    let data = svcreate4_s8(
        svindex_s8((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_s8((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_s8((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_s8((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    svst4_s8(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld4_s8(svptrue_b8(), storage.as_ptr() as *const i8);
    assert_vector_matches_i8(
        svget4_s8::<{ 0usize as i32 }>(loaded),
        svindex_s8((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_i8(
        svget4_s8::<{ 1usize as i32 }>(loaded),
        svindex_s8((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_i8(
        svget4_s8::<{ 2usize as i32 }>(loaded),
        svindex_s8((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_i8(
        svget4_s8::<{ 3usize as i32 }>(loaded),
        svindex_s8((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_s16_with_svst4_s16() {
    let mut storage = [0 as i16; 640usize];
    let data = svcreate4_s16(
        svindex_s16((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_s16((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_s16((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_s16((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    svst4_s16(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld4_s16(svptrue_b16(), storage.as_ptr() as *const i16);
    assert_vector_matches_i16(
        svget4_s16::<{ 0usize as i32 }>(loaded),
        svindex_s16((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_i16(
        svget4_s16::<{ 1usize as i32 }>(loaded),
        svindex_s16((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_i16(
        svget4_s16::<{ 2usize as i32 }>(loaded),
        svindex_s16((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_i16(
        svget4_s16::<{ 3usize as i32 }>(loaded),
        svindex_s16((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_s32_with_svst4_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svcreate4_s32(
        svindex_s32((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_s32((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_s32((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_s32((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    svst4_s32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld4_s32(svptrue_b32(), storage.as_ptr() as *const i32);
    assert_vector_matches_i32(
        svget4_s32::<{ 0usize as i32 }>(loaded),
        svindex_s32((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_i32(
        svget4_s32::<{ 1usize as i32 }>(loaded),
        svindex_s32((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_i32(
        svget4_s32::<{ 2usize as i32 }>(loaded),
        svindex_s32((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_i32(
        svget4_s32::<{ 3usize as i32 }>(loaded),
        svindex_s32((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_s64_with_svst4_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svcreate4_s64(
        svindex_s64((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_s64((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_s64((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_s64((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    svst4_s64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld4_s64(svptrue_b64(), storage.as_ptr() as *const i64);
    assert_vector_matches_i64(
        svget4_s64::<{ 0usize as i32 }>(loaded),
        svindex_s64((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_i64(
        svget4_s64::<{ 1usize as i32 }>(loaded),
        svindex_s64((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_i64(
        svget4_s64::<{ 2usize as i32 }>(loaded),
        svindex_s64((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_i64(
        svget4_s64::<{ 3usize as i32 }>(loaded),
        svindex_s64((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_u8_with_svst4_u8() {
    let mut storage = [0 as u8; 1280usize];
    let data = svcreate4_u8(
        svindex_u8((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_u8((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_u8((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_u8((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    svst4_u8(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld4_u8(svptrue_b8(), storage.as_ptr() as *const u8);
    assert_vector_matches_u8(
        svget4_u8::<{ 0usize as i32 }>(loaded),
        svindex_u8((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_u8(
        svget4_u8::<{ 1usize as i32 }>(loaded),
        svindex_u8((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_u8(
        svget4_u8::<{ 2usize as i32 }>(loaded),
        svindex_u8((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_u8(
        svget4_u8::<{ 3usize as i32 }>(loaded),
        svindex_u8((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_u16_with_svst4_u16() {
    let mut storage = [0 as u16; 640usize];
    let data = svcreate4_u16(
        svindex_u16((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_u16((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_u16((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_u16((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    svst4_u16(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld4_u16(svptrue_b16(), storage.as_ptr() as *const u16);
    assert_vector_matches_u16(
        svget4_u16::<{ 0usize as i32 }>(loaded),
        svindex_u16((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_u16(
        svget4_u16::<{ 1usize as i32 }>(loaded),
        svindex_u16((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_u16(
        svget4_u16::<{ 2usize as i32 }>(loaded),
        svindex_u16((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_u16(
        svget4_u16::<{ 3usize as i32 }>(loaded),
        svindex_u16((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_u32_with_svst4_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svcreate4_u32(
        svindex_u32((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_u32((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_u32((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_u32((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    svst4_u32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld4_u32(svptrue_b32(), storage.as_ptr() as *const u32);
    assert_vector_matches_u32(
        svget4_u32::<{ 0usize as i32 }>(loaded),
        svindex_u32((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_u32(
        svget4_u32::<{ 1usize as i32 }>(loaded),
        svindex_u32((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_u32(
        svget4_u32::<{ 2usize as i32 }>(loaded),
        svindex_u32((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_u32(
        svget4_u32::<{ 3usize as i32 }>(loaded),
        svindex_u32((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_u64_with_svst4_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svcreate4_u64(
        svindex_u64((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_u64((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_u64((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
        svindex_u64((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    svst4_u64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld4_u64(svptrue_b64(), storage.as_ptr() as *const u64);
    assert_vector_matches_u64(
        svget4_u64::<{ 0usize as i32 }>(loaded),
        svindex_u64((0usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_u64(
        svget4_u64::<{ 1usize as i32 }>(loaded),
        svindex_u64((1usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_u64(
        svget4_u64::<{ 2usize as i32 }>(loaded),
        svindex_u64((2usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
    assert_vector_matches_u64(
        svget4_u64::<{ 3usize as i32 }>(loaded),
        svindex_u64((3usize).try_into().unwrap(), 4usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_vnum_f32_with_svst4_vnum_f32() {
    let len = svcntw() as usize;
    let mut storage = [0 as f32; 320usize];
    let data = svcreate4_f32(
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 0usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 1usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 2usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 3usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
    );
    svst4_vnum_f32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svld4_vnum_f32(svptrue_b32(), storage.as_ptr() as *const f32, 1);
    assert_vector_matches_f32(
        svget4_f32::<{ 0usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 0usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
    );
    assert_vector_matches_f32(
        svget4_f32::<{ 1usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 1usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
    );
    assert_vector_matches_f32(
        svget4_f32::<{ 2usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 2usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
    );
    assert_vector_matches_f32(
        svget4_f32::<{ 3usize as i32 }>(loaded),
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 3usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_vnum_f64_with_svst4_vnum_f64() {
    let len = svcntd() as usize;
    let mut storage = [0 as f64; 160usize];
    let data = svcreate4_f64(
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 0usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 1usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 2usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 3usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
    );
    svst4_vnum_f64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svld4_vnum_f64(svptrue_b64(), storage.as_ptr() as *const f64, 1);
    assert_vector_matches_f64(
        svget4_f64::<{ 0usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 0usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
    );
    assert_vector_matches_f64(
        svget4_f64::<{ 1usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 1usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
    );
    assert_vector_matches_f64(
        svget4_f64::<{ 2usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 2usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
    );
    assert_vector_matches_f64(
        svget4_f64::<{ 3usize as i32 }>(loaded),
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 3usize).try_into().unwrap(),
                4usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_vnum_s8_with_svst4_vnum_s8() {
    let len = svcntb() as usize;
    let mut storage = [0 as i8; 1280usize];
    let data = svcreate4_s8(
        svindex_s8(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_s8(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_s8(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_s8(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    svst4_vnum_s8(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svld4_vnum_s8(svptrue_b8(), storage.as_ptr() as *const i8, 1);
    assert_vector_matches_i8(
        svget4_s8::<{ 0usize as i32 }>(loaded),
        svindex_s8(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i8(
        svget4_s8::<{ 1usize as i32 }>(loaded),
        svindex_s8(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i8(
        svget4_s8::<{ 2usize as i32 }>(loaded),
        svindex_s8(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i8(
        svget4_s8::<{ 3usize as i32 }>(loaded),
        svindex_s8(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_vnum_s16_with_svst4_vnum_s16() {
    let len = svcnth() as usize;
    let mut storage = [0 as i16; 640usize];
    let data = svcreate4_s16(
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_s16(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_s16(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_s16(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    svst4_vnum_s16(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svld4_vnum_s16(svptrue_b16(), storage.as_ptr() as *const i16, 1);
    assert_vector_matches_i16(
        svget4_s16::<{ 0usize as i32 }>(loaded),
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i16(
        svget4_s16::<{ 1usize as i32 }>(loaded),
        svindex_s16(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i16(
        svget4_s16::<{ 2usize as i32 }>(loaded),
        svindex_s16(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i16(
        svget4_s16::<{ 3usize as i32 }>(loaded),
        svindex_s16(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_vnum_s32_with_svst4_vnum_s32() {
    let len = svcntw() as usize;
    let mut storage = [0 as i32; 320usize];
    let data = svcreate4_s32(
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_s32(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_s32(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_s32(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    svst4_vnum_s32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svld4_vnum_s32(svptrue_b32(), storage.as_ptr() as *const i32, 1);
    assert_vector_matches_i32(
        svget4_s32::<{ 0usize as i32 }>(loaded),
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i32(
        svget4_s32::<{ 1usize as i32 }>(loaded),
        svindex_s32(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i32(
        svget4_s32::<{ 2usize as i32 }>(loaded),
        svindex_s32(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i32(
        svget4_s32::<{ 3usize as i32 }>(loaded),
        svindex_s32(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_vnum_s64_with_svst4_vnum_s64() {
    let len = svcntd() as usize;
    let mut storage = [0 as i64; 160usize];
    let data = svcreate4_s64(
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_s64(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_s64(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_s64(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    svst4_vnum_s64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svld4_vnum_s64(svptrue_b64(), storage.as_ptr() as *const i64, 1);
    assert_vector_matches_i64(
        svget4_s64::<{ 0usize as i32 }>(loaded),
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i64(
        svget4_s64::<{ 1usize as i32 }>(loaded),
        svindex_s64(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i64(
        svget4_s64::<{ 2usize as i32 }>(loaded),
        svindex_s64(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_i64(
        svget4_s64::<{ 3usize as i32 }>(loaded),
        svindex_s64(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_vnum_u8_with_svst4_vnum_u8() {
    let len = svcntb() as usize;
    let mut storage = [0 as u8; 1280usize];
    let data = svcreate4_u8(
        svindex_u8(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_u8(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_u8(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_u8(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    svst4_vnum_u8(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svld4_vnum_u8(svptrue_b8(), storage.as_ptr() as *const u8, 1);
    assert_vector_matches_u8(
        svget4_u8::<{ 0usize as i32 }>(loaded),
        svindex_u8(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u8(
        svget4_u8::<{ 1usize as i32 }>(loaded),
        svindex_u8(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u8(
        svget4_u8::<{ 2usize as i32 }>(loaded),
        svindex_u8(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u8(
        svget4_u8::<{ 3usize as i32 }>(loaded),
        svindex_u8(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_vnum_u16_with_svst4_vnum_u16() {
    let len = svcnth() as usize;
    let mut storage = [0 as u16; 640usize];
    let data = svcreate4_u16(
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_u16(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_u16(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_u16(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    svst4_vnum_u16(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svld4_vnum_u16(svptrue_b16(), storage.as_ptr() as *const u16, 1);
    assert_vector_matches_u16(
        svget4_u16::<{ 0usize as i32 }>(loaded),
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u16(
        svget4_u16::<{ 1usize as i32 }>(loaded),
        svindex_u16(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u16(
        svget4_u16::<{ 2usize as i32 }>(loaded),
        svindex_u16(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u16(
        svget4_u16::<{ 3usize as i32 }>(loaded),
        svindex_u16(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_vnum_u32_with_svst4_vnum_u32() {
    let len = svcntw() as usize;
    let mut storage = [0 as u32; 320usize];
    let data = svcreate4_u32(
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_u32(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_u32(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_u32(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    svst4_vnum_u32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svld4_vnum_u32(svptrue_b32(), storage.as_ptr() as *const u32, 1);
    assert_vector_matches_u32(
        svget4_u32::<{ 0usize as i32 }>(loaded),
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u32(
        svget4_u32::<{ 1usize as i32 }>(loaded),
        svindex_u32(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u32(
        svget4_u32::<{ 2usize as i32 }>(loaded),
        svindex_u32(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u32(
        svget4_u32::<{ 3usize as i32 }>(loaded),
        svindex_u32(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svld4_vnum_u64_with_svst4_vnum_u64() {
    let len = svcntd() as usize;
    let mut storage = [0 as u64; 160usize];
    let data = svcreate4_u64(
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_u64(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_u64(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
        svindex_u64(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    svst4_vnum_u64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svld4_vnum_u64(svptrue_b64(), storage.as_ptr() as *const u64, 1);
    assert_vector_matches_u64(
        svget4_u64::<{ 0usize as i32 }>(loaded),
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u64(
        svget4_u64::<{ 1usize as i32 }>(loaded),
        svindex_u64(
            (len + 1usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u64(
        svget4_u64::<{ 2usize as i32 }>(loaded),
        svindex_u64(
            (len + 2usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
    assert_vector_matches_u64(
        svget4_u64::<{ 3usize as i32 }>(loaded),
        svindex_u64(
            (len + 3usize).try_into().unwrap(),
            4usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_f32() {
    svsetffr();
    let _ = svld1_f32(svptrue_b32(), F32_DATA.as_ptr());
    let loaded = svldff1_f32(svptrue_b32(), F32_DATA.as_ptr());
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_f64() {
    svsetffr();
    let _ = svld1_f64(svptrue_b64(), F64_DATA.as_ptr());
    let loaded = svldff1_f64(svptrue_b64(), F64_DATA.as_ptr());
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_s8() {
    svsetffr();
    let _ = svld1_s8(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldff1_s8(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_i8(
        loaded,
        svindex_s8((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_s16() {
    svsetffr();
    let _ = svld1_s16(svptrue_b16(), I16_DATA.as_ptr());
    let loaded = svldff1_s16(svptrue_b16(), I16_DATA.as_ptr());
    assert_vector_matches_i16(
        loaded,
        svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_s32() {
    svsetffr();
    let _ = svld1_s32(svptrue_b32(), I32_DATA.as_ptr());
    let loaded = svldff1_s32(svptrue_b32(), I32_DATA.as_ptr());
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_s64() {
    svsetffr();
    let _ = svld1_s64(svptrue_b64(), I64_DATA.as_ptr());
    let loaded = svldff1_s64(svptrue_b64(), I64_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_u8() {
    svsetffr();
    let _ = svld1_u8(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldff1_u8(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_u8(
        loaded,
        svindex_u8((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_u16() {
    svsetffr();
    let _ = svld1_u16(svptrue_b16(), U16_DATA.as_ptr());
    let loaded = svldff1_u16(svptrue_b16(), U16_DATA.as_ptr());
    assert_vector_matches_u16(
        loaded,
        svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_u32() {
    svsetffr();
    let _ = svld1_u32(svptrue_b32(), U32_DATA.as_ptr());
    let loaded = svldff1_u32(svptrue_b32(), U32_DATA.as_ptr());
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_u64() {
    svsetffr();
    let _ = svld1_u64(svptrue_b64(), U64_DATA.as_ptr());
    let loaded = svldff1_u64(svptrue_b64(), U64_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_s32index_f32() {
    let indices = svindex_s32(0, 1);
    svsetffr();
    let _ = svld1_gather_s32index_f32(svptrue_b32(), F32_DATA.as_ptr(), indices);
    let loaded = svldff1_gather_s32index_f32(svptrue_b32(), F32_DATA.as_ptr(), indices);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_s32index_s32() {
    let indices = svindex_s32(0, 1);
    svsetffr();
    let _ = svld1_gather_s32index_s32(svptrue_b32(), I32_DATA.as_ptr(), indices);
    let loaded = svldff1_gather_s32index_s32(svptrue_b32(), I32_DATA.as_ptr(), indices);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_s32index_u32() {
    let indices = svindex_s32(0, 1);
    svsetffr();
    let _ = svld1_gather_s32index_u32(svptrue_b32(), U32_DATA.as_ptr(), indices);
    let loaded = svldff1_gather_s32index_u32(svptrue_b32(), U32_DATA.as_ptr(), indices);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_s64index_f64() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let _ = svld1_gather_s64index_f64(svptrue_b64(), F64_DATA.as_ptr(), indices);
    let loaded = svldff1_gather_s64index_f64(svptrue_b64(), F64_DATA.as_ptr(), indices);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_s64index_s64() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let _ = svld1_gather_s64index_s64(svptrue_b64(), I64_DATA.as_ptr(), indices);
    let loaded = svldff1_gather_s64index_s64(svptrue_b64(), I64_DATA.as_ptr(), indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_s64index_u64() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let _ = svld1_gather_s64index_u64(svptrue_b64(), U64_DATA.as_ptr(), indices);
    let loaded = svldff1_gather_s64index_u64(svptrue_b64(), U64_DATA.as_ptr(), indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u32index_f32() {
    let indices = svindex_u32(0, 1);
    svsetffr();
    let _ = svld1_gather_u32index_f32(svptrue_b32(), F32_DATA.as_ptr(), indices);
    let loaded = svldff1_gather_u32index_f32(svptrue_b32(), F32_DATA.as_ptr(), indices);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u32index_s32() {
    let indices = svindex_u32(0, 1);
    svsetffr();
    let _ = svld1_gather_u32index_s32(svptrue_b32(), I32_DATA.as_ptr(), indices);
    let loaded = svldff1_gather_u32index_s32(svptrue_b32(), I32_DATA.as_ptr(), indices);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u32index_u32() {
    let indices = svindex_u32(0, 1);
    svsetffr();
    let _ = svld1_gather_u32index_u32(svptrue_b32(), U32_DATA.as_ptr(), indices);
    let loaded = svldff1_gather_u32index_u32(svptrue_b32(), U32_DATA.as_ptr(), indices);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64index_f64() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let _ = svld1_gather_u64index_f64(svptrue_b64(), F64_DATA.as_ptr(), indices);
    let loaded = svldff1_gather_u64index_f64(svptrue_b64(), F64_DATA.as_ptr(), indices);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64index_s64() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let _ = svld1_gather_u64index_s64(svptrue_b64(), I64_DATA.as_ptr(), indices);
    let loaded = svldff1_gather_u64index_s64(svptrue_b64(), I64_DATA.as_ptr(), indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64index_u64() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let _ = svld1_gather_u64index_u64(svptrue_b64(), U64_DATA.as_ptr(), indices);
    let loaded = svldff1_gather_u64index_u64(svptrue_b64(), U64_DATA.as_ptr(), indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_s32offset_f32() {
    let offsets = svindex_s32(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_s32offset_f32(svptrue_b32(), F32_DATA.as_ptr(), offsets);
    let loaded = svldff1_gather_s32offset_f32(svptrue_b32(), F32_DATA.as_ptr(), offsets);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_s32offset_s32() {
    let offsets = svindex_s32(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_s32offset_s32(svptrue_b32(), I32_DATA.as_ptr(), offsets);
    let loaded = svldff1_gather_s32offset_s32(svptrue_b32(), I32_DATA.as_ptr(), offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_s32offset_u32() {
    let offsets = svindex_s32(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_s32offset_u32(svptrue_b32(), U32_DATA.as_ptr(), offsets);
    let loaded = svldff1_gather_s32offset_u32(svptrue_b32(), U32_DATA.as_ptr(), offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_s64offset_f64() {
    let offsets = svindex_s64(0, 8u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_s64offset_f64(svptrue_b64(), F64_DATA.as_ptr(), offsets);
    let loaded = svldff1_gather_s64offset_f64(svptrue_b64(), F64_DATA.as_ptr(), offsets);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_s64offset_s64() {
    let offsets = svindex_s64(0, 8u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_s64offset_s64(svptrue_b64(), I64_DATA.as_ptr(), offsets);
    let loaded = svldff1_gather_s64offset_s64(svptrue_b64(), I64_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_s64offset_u64() {
    let offsets = svindex_s64(0, 8u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_s64offset_u64(svptrue_b64(), U64_DATA.as_ptr(), offsets);
    let loaded = svldff1_gather_s64offset_u64(svptrue_b64(), U64_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u32offset_f32() {
    let offsets = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_u32offset_f32(svptrue_b32(), F32_DATA.as_ptr(), offsets);
    let loaded = svldff1_gather_u32offset_f32(svptrue_b32(), F32_DATA.as_ptr(), offsets);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u32offset_s32() {
    let offsets = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_u32offset_s32(svptrue_b32(), I32_DATA.as_ptr(), offsets);
    let loaded = svldff1_gather_u32offset_s32(svptrue_b32(), I32_DATA.as_ptr(), offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u32offset_u32() {
    let offsets = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_u32offset_u32(svptrue_b32(), U32_DATA.as_ptr(), offsets);
    let loaded = svldff1_gather_u32offset_u32(svptrue_b32(), U32_DATA.as_ptr(), offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64offset_f64() {
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_u64offset_f64(svptrue_b64(), F64_DATA.as_ptr(), offsets);
    let loaded = svldff1_gather_u64offset_f64(svptrue_b64(), F64_DATA.as_ptr(), offsets);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64offset_s64() {
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_u64offset_s64(svptrue_b64(), I64_DATA.as_ptr(), offsets);
    let loaded = svldff1_gather_u64offset_s64(svptrue_b64(), I64_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64offset_u64() {
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_u64offset_u64(svptrue_b64(), U64_DATA.as_ptr(), offsets);
    let loaded = svldff1_gather_u64offset_u64(svptrue_b64(), U64_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64base_f64() {
    let bases = svdup_n_u64(F64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let _ = svld1_gather_u64base_f64(svptrue_b64(), bases);
    let loaded = svldff1_gather_u64base_f64(svptrue_b64(), bases);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64base_s64() {
    let bases = svdup_n_u64(I64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let _ = svld1_gather_u64base_s64(svptrue_b64(), bases);
    let loaded = svldff1_gather_u64base_s64(svptrue_b64(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64base_u64() {
    let bases = svdup_n_u64(U64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let _ = svld1_gather_u64base_u64(svptrue_b64(), bases);
    let loaded = svldff1_gather_u64base_u64(svptrue_b64(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u32base_index_f32() {
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_u32base_index_f32(
        svptrue_b32(),
        bases,
        F32_DATA.as_ptr() as i64 / (4u32 as i64) + 1,
    );
    let loaded = svldff1_gather_u32base_index_f32(
        svptrue_b32(),
        bases,
        F32_DATA.as_ptr() as i64 / (4u32 as i64) + 1,
    );
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u32base_index_s32() {
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_u32base_index_s32(
        svptrue_b32(),
        bases,
        I32_DATA.as_ptr() as i64 / (4u32 as i64) + 1,
    );
    let loaded = svldff1_gather_u32base_index_s32(
        svptrue_b32(),
        bases,
        I32_DATA.as_ptr() as i64 / (4u32 as i64) + 1,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u32base_index_u32() {
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_u32base_index_u32(
        svptrue_b32(),
        bases,
        U32_DATA.as_ptr() as i64 / (4u32 as i64) + 1,
    );
    let loaded = svldff1_gather_u32base_index_u32(
        svptrue_b32(),
        bases,
        U32_DATA.as_ptr() as i64 / (4u32 as i64) + 1,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64base_index_f64() {
    let bases = svdup_n_u64(F64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let _ = svld1_gather_u64base_index_f64(svptrue_b64(), bases, 1.try_into().unwrap());
    let loaded = svldff1_gather_u64base_index_f64(svptrue_b64(), bases, 1.try_into().unwrap());
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64base_index_s64() {
    let bases = svdup_n_u64(I64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let _ = svld1_gather_u64base_index_s64(svptrue_b64(), bases, 1.try_into().unwrap());
    let loaded = svldff1_gather_u64base_index_s64(svptrue_b64(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64base_index_u64() {
    let bases = svdup_n_u64(U64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let _ = svld1_gather_u64base_index_u64(svptrue_b64(), bases, 1.try_into().unwrap());
    let loaded = svldff1_gather_u64base_index_u64(svptrue_b64(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u32base_offset_f32() {
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_u32base_offset_f32(
        svptrue_b32(),
        bases,
        F32_DATA.as_ptr() as i64 + 4u32 as i64,
    );
    let loaded = svldff1_gather_u32base_offset_f32(
        svptrue_b32(),
        bases,
        F32_DATA.as_ptr() as i64 + 4u32 as i64,
    );
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u32base_offset_s32() {
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_u32base_offset_s32(
        svptrue_b32(),
        bases,
        I32_DATA.as_ptr() as i64 + 4u32 as i64,
    );
    let loaded = svldff1_gather_u32base_offset_s32(
        svptrue_b32(),
        bases,
        I32_DATA.as_ptr() as i64 + 4u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u32base_offset_u32() {
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1_gather_u32base_offset_u32(
        svptrue_b32(),
        bases,
        U32_DATA.as_ptr() as i64 + 4u32 as i64,
    );
    let loaded = svldff1_gather_u32base_offset_u32(
        svptrue_b32(),
        bases,
        U32_DATA.as_ptr() as i64 + 4u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64base_offset_f64() {
    let bases = svdup_n_u64(F64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let _ = svld1_gather_u64base_offset_f64(svptrue_b64(), bases, 8u32.try_into().unwrap());
    let loaded = svldff1_gather_u64base_offset_f64(svptrue_b64(), bases, 8u32.try_into().unwrap());
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64base_offset_s64() {
    let bases = svdup_n_u64(I64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let _ = svld1_gather_u64base_offset_s64(svptrue_b64(), bases, 8u32.try_into().unwrap());
    let loaded = svldff1_gather_u64base_offset_s64(svptrue_b64(), bases, 8u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_gather_u64base_offset_u64() {
    let bases = svdup_n_u64(U64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let _ = svld1_gather_u64base_offset_u64(svptrue_b64(), bases, 8u32.try_into().unwrap());
    let loaded = svldff1_gather_u64base_offset_u64(svptrue_b64(), bases, 8u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_vnum_f32() {
    svsetffr();
    let _ = svld1_vnum_f32(svptrue_b32(), F32_DATA.as_ptr(), 1);
    let loaded = svldff1_vnum_f32(svptrue_b32(), F32_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 0usize).try_into().unwrap(),
                1usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_vnum_f64() {
    svsetffr();
    let _ = svld1_vnum_f64(svptrue_b64(), F64_DATA.as_ptr(), 1);
    let loaded = svldff1_vnum_f64(svptrue_b64(), F64_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 0usize).try_into().unwrap(),
                1usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_vnum_s8() {
    svsetffr();
    let _ = svld1_vnum_s8(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldff1_vnum_s8(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcntb() as usize;
    assert_vector_matches_i8(
        loaded,
        svindex_s8(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_vnum_s16() {
    svsetffr();
    let _ = svld1_vnum_s16(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let loaded = svldff1_vnum_s16(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let len = svcnth() as usize;
    assert_vector_matches_i16(
        loaded,
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_vnum_s32() {
    svsetffr();
    let _ = svld1_vnum_s32(svptrue_b32(), I32_DATA.as_ptr(), 1);
    let loaded = svldff1_vnum_s32(svptrue_b32(), I32_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_vnum_s64() {
    svsetffr();
    let _ = svld1_vnum_s64(svptrue_b64(), I64_DATA.as_ptr(), 1);
    let loaded = svldff1_vnum_s64(svptrue_b64(), I64_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_vnum_u8() {
    svsetffr();
    let _ = svld1_vnum_u8(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldff1_vnum_u8(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcntb() as usize;
    assert_vector_matches_u8(
        loaded,
        svindex_u8(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_vnum_u16() {
    svsetffr();
    let _ = svld1_vnum_u16(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let loaded = svldff1_vnum_u16(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let len = svcnth() as usize;
    assert_vector_matches_u16(
        loaded,
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_vnum_u32() {
    svsetffr();
    let _ = svld1_vnum_u32(svptrue_b32(), U32_DATA.as_ptr(), 1);
    let loaded = svldff1_vnum_u32(svptrue_b32(), U32_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1_vnum_u64() {
    svsetffr();
    let _ = svld1_vnum_u64(svptrue_b64(), U64_DATA.as_ptr(), 1);
    let loaded = svldff1_vnum_u64(svptrue_b64(), U64_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_s32offset_s32() {
    let offsets = svindex_s32(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sb_gather_s32offset_s32(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    let loaded = svldff1sb_gather_s32offset_s32(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_s32offset_s32() {
    let offsets = svindex_s32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sh_gather_s32offset_s32(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    let loaded = svldff1sh_gather_s32offset_s32(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_s32offset_u32() {
    let offsets = svindex_s32(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sb_gather_s32offset_u32(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    let loaded = svldff1sb_gather_s32offset_u32(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_s32offset_u32() {
    let offsets = svindex_s32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sh_gather_s32offset_u32(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    let loaded = svldff1sh_gather_s32offset_u32(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_s64offset_s64() {
    let offsets = svindex_s64(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sb_gather_s64offset_s64(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    let loaded = svldff1sb_gather_s64offset_s64(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_s64offset_s64() {
    let offsets = svindex_s64(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sh_gather_s64offset_s64(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    let loaded = svldff1sh_gather_s64offset_s64(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_s64offset_s64() {
    let offsets = svindex_s64(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sw_gather_s64offset_s64(svptrue_b32(), I32_DATA.as_ptr(), offsets);
    let loaded = svldff1sw_gather_s64offset_s64(svptrue_b32(), I32_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_s64offset_u64() {
    let offsets = svindex_s64(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sb_gather_s64offset_u64(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    let loaded = svldff1sb_gather_s64offset_u64(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_s64offset_u64() {
    let offsets = svindex_s64(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sh_gather_s64offset_u64(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    let loaded = svldff1sh_gather_s64offset_u64(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_s64offset_u64() {
    let offsets = svindex_s64(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sw_gather_s64offset_u64(svptrue_b32(), I32_DATA.as_ptr(), offsets);
    let loaded = svldff1sw_gather_s64offset_u64(svptrue_b32(), I32_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_u32offset_s32() {
    let offsets = svindex_u32(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sb_gather_u32offset_s32(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    let loaded = svldff1sb_gather_u32offset_s32(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u32offset_s32() {
    let offsets = svindex_u32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sh_gather_u32offset_s32(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    let loaded = svldff1sh_gather_u32offset_s32(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_u32offset_u32() {
    let offsets = svindex_u32(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sb_gather_u32offset_u32(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    let loaded = svldff1sb_gather_u32offset_u32(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u32offset_u32() {
    let offsets = svindex_u32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sh_gather_u32offset_u32(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    let loaded = svldff1sh_gather_u32offset_u32(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_u64offset_s64() {
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sb_gather_u64offset_s64(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    let loaded = svldff1sb_gather_u64offset_s64(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u64offset_s64() {
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sh_gather_u64offset_s64(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    let loaded = svldff1sh_gather_u64offset_s64(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_u64offset_s64() {
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sw_gather_u64offset_s64(svptrue_b32(), I32_DATA.as_ptr(), offsets);
    let loaded = svldff1sw_gather_u64offset_s64(svptrue_b32(), I32_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_u64offset_u64() {
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sb_gather_u64offset_u64(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    let loaded = svldff1sb_gather_u64offset_u64(svptrue_b8(), I8_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u64offset_u64() {
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sh_gather_u64offset_u64(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    let loaded = svldff1sh_gather_u64offset_u64(svptrue_b16(), I16_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_u64offset_u64() {
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sw_gather_u64offset_u64(svptrue_b32(), I32_DATA.as_ptr(), offsets);
    let loaded = svldff1sw_gather_u64offset_u64(svptrue_b32(), I32_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_u32base_offset_s32() {
    let bases = svindex_u32(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sb_gather_u32base_offset_s32(
        svptrue_b8(),
        bases,
        I8_DATA.as_ptr() as i64 + 1u32 as i64,
    );
    let loaded = svldff1sb_gather_u32base_offset_s32(
        svptrue_b8(),
        bases,
        I8_DATA.as_ptr() as i64 + 1u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u32base_offset_s32() {
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sh_gather_u32base_offset_s32(
        svptrue_b16(),
        bases,
        I16_DATA.as_ptr() as i64 + 2u32 as i64,
    );
    let loaded = svldff1sh_gather_u32base_offset_s32(
        svptrue_b16(),
        bases,
        I16_DATA.as_ptr() as i64 + 2u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_u32base_offset_u32() {
    let bases = svindex_u32(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sb_gather_u32base_offset_u32(
        svptrue_b8(),
        bases,
        I8_DATA.as_ptr() as i64 + 1u32 as i64,
    );
    let loaded = svldff1sb_gather_u32base_offset_u32(
        svptrue_b8(),
        bases,
        I8_DATA.as_ptr() as i64 + 1u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u32base_offset_u32() {
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sh_gather_u32base_offset_u32(
        svptrue_b16(),
        bases,
        I16_DATA.as_ptr() as i64 + 2u32 as i64,
    );
    let loaded = svldff1sh_gather_u32base_offset_u32(
        svptrue_b16(),
        bases,
        I16_DATA.as_ptr() as i64 + 2u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_u64base_offset_s64() {
    let bases = svdup_n_u64(I8_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svsetffr();
    let _ = svld1sb_gather_u64base_offset_s64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    let loaded = svldff1sb_gather_u64base_offset_s64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u64base_offset_s64() {
    let bases = svdup_n_u64(I16_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svsetffr();
    let _ = svld1sh_gather_u64base_offset_s64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    let loaded =
        svldff1sh_gather_u64base_offset_s64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_u64base_offset_s64() {
    let bases = svdup_n_u64(I32_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svsetffr();
    let _ = svld1sw_gather_u64base_offset_s64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    let loaded =
        svldff1sw_gather_u64base_offset_s64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_u64base_offset_u64() {
    let bases = svdup_n_u64(I8_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svsetffr();
    let _ = svld1sb_gather_u64base_offset_u64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    let loaded = svldff1sb_gather_u64base_offset_u64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u64base_offset_u64() {
    let bases = svdup_n_u64(I16_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svsetffr();
    let _ = svld1sh_gather_u64base_offset_u64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    let loaded =
        svldff1sh_gather_u64base_offset_u64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_u64base_offset_u64() {
    let bases = svdup_n_u64(I32_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svsetffr();
    let _ = svld1sw_gather_u64base_offset_u64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    let loaded =
        svldff1sw_gather_u64base_offset_u64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_u64base_s64() {
    let bases = svdup_n_u64(I8_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svsetffr();
    let _ = svld1sb_gather_u64base_s64(svptrue_b8(), bases);
    let loaded = svldff1sb_gather_u64base_s64(svptrue_b8(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u64base_s64() {
    let bases = svdup_n_u64(I16_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svsetffr();
    let _ = svld1sh_gather_u64base_s64(svptrue_b16(), bases);
    let loaded = svldff1sh_gather_u64base_s64(svptrue_b16(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_u64base_s64() {
    let bases = svdup_n_u64(I32_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svsetffr();
    let _ = svld1sw_gather_u64base_s64(svptrue_b32(), bases);
    let loaded = svldff1sw_gather_u64base_s64(svptrue_b32(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_gather_u64base_u64() {
    let bases = svdup_n_u64(I8_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svsetffr();
    let _ = svld1sb_gather_u64base_u64(svptrue_b8(), bases);
    let loaded = svldff1sb_gather_u64base_u64(svptrue_b8(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u64base_u64() {
    let bases = svdup_n_u64(I16_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svsetffr();
    let _ = svld1sh_gather_u64base_u64(svptrue_b16(), bases);
    let loaded = svldff1sh_gather_u64base_u64(svptrue_b16(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_u64base_u64() {
    let bases = svdup_n_u64(I32_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svsetffr();
    let _ = svld1sw_gather_u64base_u64(svptrue_b32(), bases);
    let loaded = svldff1sw_gather_u64base_u64(svptrue_b32(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_s16() {
    svsetffr();
    let _ = svld1sb_s16(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldff1sb_s16(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_i16(
        loaded,
        svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_s32() {
    svsetffr();
    let _ = svld1sb_s32(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldff1sb_s32(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_s32() {
    svsetffr();
    let _ = svld1sh_s32(svptrue_b16(), I16_DATA.as_ptr());
    let loaded = svldff1sh_s32(svptrue_b16(), I16_DATA.as_ptr());
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_s64() {
    svsetffr();
    let _ = svld1sb_s64(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldff1sb_s64(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_s64() {
    svsetffr();
    let _ = svld1sh_s64(svptrue_b16(), I16_DATA.as_ptr());
    let loaded = svldff1sh_s64(svptrue_b16(), I16_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_s64() {
    svsetffr();
    let _ = svld1sw_s64(svptrue_b32(), I32_DATA.as_ptr());
    let loaded = svldff1sw_s64(svptrue_b32(), I32_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_u16() {
    svsetffr();
    let _ = svld1sb_u16(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldff1sb_u16(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_u16(
        loaded,
        svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_u32() {
    svsetffr();
    let _ = svld1sb_u32(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldff1sb_u32(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_u32() {
    svsetffr();
    let _ = svld1sh_u32(svptrue_b16(), I16_DATA.as_ptr());
    let loaded = svldff1sh_u32(svptrue_b16(), I16_DATA.as_ptr());
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_u64() {
    svsetffr();
    let _ = svld1sb_u64(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldff1sb_u64(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_u64() {
    svsetffr();
    let _ = svld1sh_u64(svptrue_b16(), I16_DATA.as_ptr());
    let loaded = svldff1sh_u64(svptrue_b16(), I16_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_u64() {
    svsetffr();
    let _ = svld1sw_u64(svptrue_b32(), I32_DATA.as_ptr());
    let loaded = svldff1sw_u64(svptrue_b32(), I32_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_vnum_s16() {
    svsetffr();
    let _ = svld1sb_vnum_s16(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldff1sb_vnum_s16(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcnth() as usize;
    assert_vector_matches_i16(
        loaded,
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_vnum_s32() {
    svsetffr();
    let _ = svld1sb_vnum_s32(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldff1sb_vnum_s32(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_vnum_s32() {
    svsetffr();
    let _ = svld1sh_vnum_s32(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let loaded = svldff1sh_vnum_s32(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_vnum_s64() {
    svsetffr();
    let _ = svld1sb_vnum_s64(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldff1sb_vnum_s64(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_vnum_s64() {
    svsetffr();
    let _ = svld1sh_vnum_s64(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let loaded = svldff1sh_vnum_s64(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_vnum_s64() {
    svsetffr();
    let _ = svld1sw_vnum_s64(svptrue_b32(), I32_DATA.as_ptr(), 1);
    let loaded = svldff1sw_vnum_s64(svptrue_b32(), I32_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_vnum_u16() {
    svsetffr();
    let _ = svld1sb_vnum_u16(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldff1sb_vnum_u16(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcnth() as usize;
    assert_vector_matches_u16(
        loaded,
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_vnum_u32() {
    svsetffr();
    let _ = svld1sb_vnum_u32(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldff1sb_vnum_u32(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_vnum_u32() {
    svsetffr();
    let _ = svld1sh_vnum_u32(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let loaded = svldff1sh_vnum_u32(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sb_vnum_u64() {
    svsetffr();
    let _ = svld1sb_vnum_u64(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldff1sb_vnum_u64(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_vnum_u64() {
    svsetffr();
    let _ = svld1sh_vnum_u64(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let loaded = svldff1sh_vnum_u64(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_vnum_u64() {
    svsetffr();
    let _ = svld1sw_vnum_u64(svptrue_b32(), I32_DATA.as_ptr(), 1);
    let loaded = svldff1sw_vnum_u64(svptrue_b32(), I32_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_s32index_s32() {
    let indices = svindex_s32(0, 1);
    svsetffr();
    let _ = svld1sh_gather_s32index_s32(svptrue_b16(), I16_DATA.as_ptr(), indices);
    let loaded = svldff1sh_gather_s32index_s32(svptrue_b16(), I16_DATA.as_ptr(), indices);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_s32index_u32() {
    let indices = svindex_s32(0, 1);
    svsetffr();
    let _ = svld1sh_gather_s32index_u32(svptrue_b16(), I16_DATA.as_ptr(), indices);
    let loaded = svldff1sh_gather_s32index_u32(svptrue_b16(), I16_DATA.as_ptr(), indices);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_s64index_s64() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let _ = svld1sh_gather_s64index_s64(svptrue_b16(), I16_DATA.as_ptr(), indices);
    let loaded = svldff1sh_gather_s64index_s64(svptrue_b16(), I16_DATA.as_ptr(), indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_s64index_s64() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let _ = svld1sw_gather_s64index_s64(svptrue_b32(), I32_DATA.as_ptr(), indices);
    let loaded = svldff1sw_gather_s64index_s64(svptrue_b32(), I32_DATA.as_ptr(), indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_s64index_u64() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let _ = svld1sh_gather_s64index_u64(svptrue_b16(), I16_DATA.as_ptr(), indices);
    let loaded = svldff1sh_gather_s64index_u64(svptrue_b16(), I16_DATA.as_ptr(), indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_s64index_u64() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let _ = svld1sw_gather_s64index_u64(svptrue_b32(), I32_DATA.as_ptr(), indices);
    let loaded = svldff1sw_gather_s64index_u64(svptrue_b32(), I32_DATA.as_ptr(), indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u32index_s32() {
    let indices = svindex_u32(0, 1);
    svsetffr();
    let _ = svld1sh_gather_u32index_s32(svptrue_b16(), I16_DATA.as_ptr(), indices);
    let loaded = svldff1sh_gather_u32index_s32(svptrue_b16(), I16_DATA.as_ptr(), indices);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u32index_u32() {
    let indices = svindex_u32(0, 1);
    svsetffr();
    let _ = svld1sh_gather_u32index_u32(svptrue_b16(), I16_DATA.as_ptr(), indices);
    let loaded = svldff1sh_gather_u32index_u32(svptrue_b16(), I16_DATA.as_ptr(), indices);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u64index_s64() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let _ = svld1sh_gather_u64index_s64(svptrue_b16(), I16_DATA.as_ptr(), indices);
    let loaded = svldff1sh_gather_u64index_s64(svptrue_b16(), I16_DATA.as_ptr(), indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_u64index_s64() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let _ = svld1sw_gather_u64index_s64(svptrue_b32(), I32_DATA.as_ptr(), indices);
    let loaded = svldff1sw_gather_u64index_s64(svptrue_b32(), I32_DATA.as_ptr(), indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u64index_u64() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let _ = svld1sh_gather_u64index_u64(svptrue_b16(), I16_DATA.as_ptr(), indices);
    let loaded = svldff1sh_gather_u64index_u64(svptrue_b16(), I16_DATA.as_ptr(), indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_u64index_u64() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let _ = svld1sw_gather_u64index_u64(svptrue_b32(), I32_DATA.as_ptr(), indices);
    let loaded = svldff1sw_gather_u64index_u64(svptrue_b32(), I32_DATA.as_ptr(), indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u32base_index_s32() {
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sh_gather_u32base_index_s32(
        svptrue_b16(),
        bases,
        I16_DATA.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    let loaded = svldff1sh_gather_u32base_index_s32(
        svptrue_b16(),
        bases,
        I16_DATA.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u32base_index_u32() {
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1sh_gather_u32base_index_u32(
        svptrue_b16(),
        bases,
        I16_DATA.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    let loaded = svldff1sh_gather_u32base_index_u32(
        svptrue_b16(),
        bases,
        I16_DATA.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u64base_index_s64() {
    let bases = svdup_n_u64(I16_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svsetffr();
    let _ = svld1sh_gather_u64base_index_s64(svptrue_b16(), bases, 1.try_into().unwrap());
    let loaded = svldff1sh_gather_u64base_index_s64(svptrue_b16(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_u64base_index_s64() {
    let bases = svdup_n_u64(I32_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svsetffr();
    let _ = svld1sw_gather_u64base_index_s64(svptrue_b32(), bases, 1.try_into().unwrap());
    let loaded = svldff1sw_gather_u64base_index_s64(svptrue_b32(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sh_gather_u64base_index_u64() {
    let bases = svdup_n_u64(I16_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svsetffr();
    let _ = svld1sh_gather_u64base_index_u64(svptrue_b16(), bases, 1.try_into().unwrap());
    let loaded = svldff1sh_gather_u64base_index_u64(svptrue_b16(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1sw_gather_u64base_index_u64() {
    let bases = svdup_n_u64(I32_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svsetffr();
    let _ = svld1sw_gather_u64base_index_u64(svptrue_b32(), bases, 1.try_into().unwrap());
    let loaded = svldff1sw_gather_u64base_index_u64(svptrue_b32(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_s32offset_s32() {
    let offsets = svindex_s32(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1ub_gather_s32offset_s32(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    let loaded = svldff1ub_gather_s32offset_s32(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_s32offset_s32() {
    let offsets = svindex_s32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uh_gather_s32offset_s32(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    let loaded = svldff1uh_gather_s32offset_s32(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_s32offset_u32() {
    let offsets = svindex_s32(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1ub_gather_s32offset_u32(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    let loaded = svldff1ub_gather_s32offset_u32(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_s32offset_u32() {
    let offsets = svindex_s32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uh_gather_s32offset_u32(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    let loaded = svldff1uh_gather_s32offset_u32(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_s64offset_s64() {
    let offsets = svindex_s64(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1ub_gather_s64offset_s64(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    let loaded = svldff1ub_gather_s64offset_s64(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_s64offset_s64() {
    let offsets = svindex_s64(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uh_gather_s64offset_s64(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    let loaded = svldff1uh_gather_s64offset_s64(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_s64offset_s64() {
    let offsets = svindex_s64(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uw_gather_s64offset_s64(svptrue_b32(), U32_DATA.as_ptr(), offsets);
    let loaded = svldff1uw_gather_s64offset_s64(svptrue_b32(), U32_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_s64offset_u64() {
    let offsets = svindex_s64(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1ub_gather_s64offset_u64(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    let loaded = svldff1ub_gather_s64offset_u64(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_s64offset_u64() {
    let offsets = svindex_s64(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uh_gather_s64offset_u64(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    let loaded = svldff1uh_gather_s64offset_u64(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_s64offset_u64() {
    let offsets = svindex_s64(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uw_gather_s64offset_u64(svptrue_b32(), U32_DATA.as_ptr(), offsets);
    let loaded = svldff1uw_gather_s64offset_u64(svptrue_b32(), U32_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_u32offset_s32() {
    let offsets = svindex_u32(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1ub_gather_u32offset_s32(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    let loaded = svldff1ub_gather_u32offset_s32(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u32offset_s32() {
    let offsets = svindex_u32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uh_gather_u32offset_s32(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    let loaded = svldff1uh_gather_u32offset_s32(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_u32offset_u32() {
    let offsets = svindex_u32(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1ub_gather_u32offset_u32(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    let loaded = svldff1ub_gather_u32offset_u32(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u32offset_u32() {
    let offsets = svindex_u32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uh_gather_u32offset_u32(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    let loaded = svldff1uh_gather_u32offset_u32(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_u64offset_s64() {
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1ub_gather_u64offset_s64(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    let loaded = svldff1ub_gather_u64offset_s64(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u64offset_s64() {
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uh_gather_u64offset_s64(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    let loaded = svldff1uh_gather_u64offset_s64(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_u64offset_s64() {
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uw_gather_u64offset_s64(svptrue_b32(), U32_DATA.as_ptr(), offsets);
    let loaded = svldff1uw_gather_u64offset_s64(svptrue_b32(), U32_DATA.as_ptr(), offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_u64offset_u64() {
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1ub_gather_u64offset_u64(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    let loaded = svldff1ub_gather_u64offset_u64(svptrue_b8(), U8_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u64offset_u64() {
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uh_gather_u64offset_u64(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    let loaded = svldff1uh_gather_u64offset_u64(svptrue_b16(), U16_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_u64offset_u64() {
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uw_gather_u64offset_u64(svptrue_b32(), U32_DATA.as_ptr(), offsets);
    let loaded = svldff1uw_gather_u64offset_u64(svptrue_b32(), U32_DATA.as_ptr(), offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_u32base_offset_s32() {
    let bases = svindex_u32(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1ub_gather_u32base_offset_s32(
        svptrue_b8(),
        bases,
        U8_DATA.as_ptr() as i64 + 1u32 as i64,
    );
    let loaded = svldff1ub_gather_u32base_offset_s32(
        svptrue_b8(),
        bases,
        U8_DATA.as_ptr() as i64 + 1u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u32base_offset_s32() {
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uh_gather_u32base_offset_s32(
        svptrue_b16(),
        bases,
        U16_DATA.as_ptr() as i64 + 2u32 as i64,
    );
    let loaded = svldff1uh_gather_u32base_offset_s32(
        svptrue_b16(),
        bases,
        U16_DATA.as_ptr() as i64 + 2u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_u32base_offset_u32() {
    let bases = svindex_u32(0, 1u32.try_into().unwrap());
    svsetffr();
    let _ = svld1ub_gather_u32base_offset_u32(
        svptrue_b8(),
        bases,
        U8_DATA.as_ptr() as i64 + 1u32 as i64,
    );
    let loaded = svldff1ub_gather_u32base_offset_u32(
        svptrue_b8(),
        bases,
        U8_DATA.as_ptr() as i64 + 1u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u32base_offset_u32() {
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uh_gather_u32base_offset_u32(
        svptrue_b16(),
        bases,
        U16_DATA.as_ptr() as i64 + 2u32 as i64,
    );
    let loaded = svldff1uh_gather_u32base_offset_u32(
        svptrue_b16(),
        bases,
        U16_DATA.as_ptr() as i64 + 2u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_u64base_offset_s64() {
    let bases = svdup_n_u64(U8_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svsetffr();
    let _ = svld1ub_gather_u64base_offset_s64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    let loaded = svldff1ub_gather_u64base_offset_s64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u64base_offset_s64() {
    let bases = svdup_n_u64(U16_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svsetffr();
    let _ = svld1uh_gather_u64base_offset_s64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    let loaded =
        svldff1uh_gather_u64base_offset_s64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_u64base_offset_s64() {
    let bases = svdup_n_u64(U32_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svsetffr();
    let _ = svld1uw_gather_u64base_offset_s64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    let loaded =
        svldff1uw_gather_u64base_offset_s64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_u64base_offset_u64() {
    let bases = svdup_n_u64(U8_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svsetffr();
    let _ = svld1ub_gather_u64base_offset_u64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    let loaded = svldff1ub_gather_u64base_offset_u64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u64base_offset_u64() {
    let bases = svdup_n_u64(U16_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svsetffr();
    let _ = svld1uh_gather_u64base_offset_u64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    let loaded =
        svldff1uh_gather_u64base_offset_u64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_u64base_offset_u64() {
    let bases = svdup_n_u64(U32_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svsetffr();
    let _ = svld1uw_gather_u64base_offset_u64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    let loaded =
        svldff1uw_gather_u64base_offset_u64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_u64base_s64() {
    let bases = svdup_n_u64(U8_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svsetffr();
    let _ = svld1ub_gather_u64base_s64(svptrue_b8(), bases);
    let loaded = svldff1ub_gather_u64base_s64(svptrue_b8(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u64base_s64() {
    let bases = svdup_n_u64(U16_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svsetffr();
    let _ = svld1uh_gather_u64base_s64(svptrue_b16(), bases);
    let loaded = svldff1uh_gather_u64base_s64(svptrue_b16(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_u64base_s64() {
    let bases = svdup_n_u64(U32_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svsetffr();
    let _ = svld1uw_gather_u64base_s64(svptrue_b32(), bases);
    let loaded = svldff1uw_gather_u64base_s64(svptrue_b32(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_gather_u64base_u64() {
    let bases = svdup_n_u64(U8_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svsetffr();
    let _ = svld1ub_gather_u64base_u64(svptrue_b8(), bases);
    let loaded = svldff1ub_gather_u64base_u64(svptrue_b8(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u64base_u64() {
    let bases = svdup_n_u64(U16_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svsetffr();
    let _ = svld1uh_gather_u64base_u64(svptrue_b16(), bases);
    let loaded = svldff1uh_gather_u64base_u64(svptrue_b16(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_u64base_u64() {
    let bases = svdup_n_u64(U32_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svsetffr();
    let _ = svld1uw_gather_u64base_u64(svptrue_b32(), bases);
    let loaded = svldff1uw_gather_u64base_u64(svptrue_b32(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_s16() {
    svsetffr();
    let _ = svld1ub_s16(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldff1ub_s16(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_i16(
        loaded,
        svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_s32() {
    svsetffr();
    let _ = svld1ub_s32(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldff1ub_s32(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_s32() {
    svsetffr();
    let _ = svld1uh_s32(svptrue_b16(), U16_DATA.as_ptr());
    let loaded = svldff1uh_s32(svptrue_b16(), U16_DATA.as_ptr());
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_s64() {
    svsetffr();
    let _ = svld1ub_s64(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldff1ub_s64(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_s64() {
    svsetffr();
    let _ = svld1uh_s64(svptrue_b16(), U16_DATA.as_ptr());
    let loaded = svldff1uh_s64(svptrue_b16(), U16_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_s64() {
    svsetffr();
    let _ = svld1uw_s64(svptrue_b32(), U32_DATA.as_ptr());
    let loaded = svldff1uw_s64(svptrue_b32(), U32_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_u16() {
    svsetffr();
    let _ = svld1ub_u16(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldff1ub_u16(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_u16(
        loaded,
        svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_u32() {
    svsetffr();
    let _ = svld1ub_u32(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldff1ub_u32(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_u32() {
    svsetffr();
    let _ = svld1uh_u32(svptrue_b16(), U16_DATA.as_ptr());
    let loaded = svldff1uh_u32(svptrue_b16(), U16_DATA.as_ptr());
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_u64() {
    svsetffr();
    let _ = svld1ub_u64(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldff1ub_u64(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_u64() {
    svsetffr();
    let _ = svld1uh_u64(svptrue_b16(), U16_DATA.as_ptr());
    let loaded = svldff1uh_u64(svptrue_b16(), U16_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_u64() {
    svsetffr();
    let _ = svld1uw_u64(svptrue_b32(), U32_DATA.as_ptr());
    let loaded = svldff1uw_u64(svptrue_b32(), U32_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_vnum_s16() {
    svsetffr();
    let _ = svld1ub_vnum_s16(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldff1ub_vnum_s16(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcnth() as usize;
    assert_vector_matches_i16(
        loaded,
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_vnum_s32() {
    svsetffr();
    let _ = svld1ub_vnum_s32(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldff1ub_vnum_s32(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_vnum_s32() {
    svsetffr();
    let _ = svld1uh_vnum_s32(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let loaded = svldff1uh_vnum_s32(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_vnum_s64() {
    svsetffr();
    let _ = svld1ub_vnum_s64(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldff1ub_vnum_s64(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_vnum_s64() {
    svsetffr();
    let _ = svld1uh_vnum_s64(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let loaded = svldff1uh_vnum_s64(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_vnum_s64() {
    svsetffr();
    let _ = svld1uw_vnum_s64(svptrue_b32(), U32_DATA.as_ptr(), 1);
    let loaded = svldff1uw_vnum_s64(svptrue_b32(), U32_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_vnum_u16() {
    svsetffr();
    let _ = svld1ub_vnum_u16(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldff1ub_vnum_u16(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcnth() as usize;
    assert_vector_matches_u16(
        loaded,
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_vnum_u32() {
    svsetffr();
    let _ = svld1ub_vnum_u32(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldff1ub_vnum_u32(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_vnum_u32() {
    svsetffr();
    let _ = svld1uh_vnum_u32(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let loaded = svldff1uh_vnum_u32(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1ub_vnum_u64() {
    svsetffr();
    let _ = svld1ub_vnum_u64(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldff1ub_vnum_u64(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_vnum_u64() {
    svsetffr();
    let _ = svld1uh_vnum_u64(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let loaded = svldff1uh_vnum_u64(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_vnum_u64() {
    svsetffr();
    let _ = svld1uw_vnum_u64(svptrue_b32(), U32_DATA.as_ptr(), 1);
    let loaded = svldff1uw_vnum_u64(svptrue_b32(), U32_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_s32index_s32() {
    let indices = svindex_s32(0, 1);
    svsetffr();
    let _ = svld1uh_gather_s32index_s32(svptrue_b16(), U16_DATA.as_ptr(), indices);
    let loaded = svldff1uh_gather_s32index_s32(svptrue_b16(), U16_DATA.as_ptr(), indices);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_s32index_u32() {
    let indices = svindex_s32(0, 1);
    svsetffr();
    let _ = svld1uh_gather_s32index_u32(svptrue_b16(), U16_DATA.as_ptr(), indices);
    let loaded = svldff1uh_gather_s32index_u32(svptrue_b16(), U16_DATA.as_ptr(), indices);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_s64index_s64() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let _ = svld1uh_gather_s64index_s64(svptrue_b16(), U16_DATA.as_ptr(), indices);
    let loaded = svldff1uh_gather_s64index_s64(svptrue_b16(), U16_DATA.as_ptr(), indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_s64index_s64() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let _ = svld1uw_gather_s64index_s64(svptrue_b32(), U32_DATA.as_ptr(), indices);
    let loaded = svldff1uw_gather_s64index_s64(svptrue_b32(), U32_DATA.as_ptr(), indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_s64index_u64() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let _ = svld1uh_gather_s64index_u64(svptrue_b16(), U16_DATA.as_ptr(), indices);
    let loaded = svldff1uh_gather_s64index_u64(svptrue_b16(), U16_DATA.as_ptr(), indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_s64index_u64() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let _ = svld1uw_gather_s64index_u64(svptrue_b32(), U32_DATA.as_ptr(), indices);
    let loaded = svldff1uw_gather_s64index_u64(svptrue_b32(), U32_DATA.as_ptr(), indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u32index_s32() {
    let indices = svindex_u32(0, 1);
    svsetffr();
    let _ = svld1uh_gather_u32index_s32(svptrue_b16(), U16_DATA.as_ptr(), indices);
    let loaded = svldff1uh_gather_u32index_s32(svptrue_b16(), U16_DATA.as_ptr(), indices);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u32index_u32() {
    let indices = svindex_u32(0, 1);
    svsetffr();
    let _ = svld1uh_gather_u32index_u32(svptrue_b16(), U16_DATA.as_ptr(), indices);
    let loaded = svldff1uh_gather_u32index_u32(svptrue_b16(), U16_DATA.as_ptr(), indices);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u64index_s64() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let _ = svld1uh_gather_u64index_s64(svptrue_b16(), U16_DATA.as_ptr(), indices);
    let loaded = svldff1uh_gather_u64index_s64(svptrue_b16(), U16_DATA.as_ptr(), indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_u64index_s64() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let _ = svld1uw_gather_u64index_s64(svptrue_b32(), U32_DATA.as_ptr(), indices);
    let loaded = svldff1uw_gather_u64index_s64(svptrue_b32(), U32_DATA.as_ptr(), indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u64index_u64() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let _ = svld1uh_gather_u64index_u64(svptrue_b16(), U16_DATA.as_ptr(), indices);
    let loaded = svldff1uh_gather_u64index_u64(svptrue_b16(), U16_DATA.as_ptr(), indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_u64index_u64() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let _ = svld1uw_gather_u64index_u64(svptrue_b32(), U32_DATA.as_ptr(), indices);
    let loaded = svldff1uw_gather_u64index_u64(svptrue_b32(), U32_DATA.as_ptr(), indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u32base_index_s32() {
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uh_gather_u32base_index_s32(
        svptrue_b16(),
        bases,
        U16_DATA.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    let loaded = svldff1uh_gather_u32base_index_s32(
        svptrue_b16(),
        bases,
        U16_DATA.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u32base_index_u32() {
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svsetffr();
    let _ = svld1uh_gather_u32base_index_u32(
        svptrue_b16(),
        bases,
        U16_DATA.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    let loaded = svldff1uh_gather_u32base_index_u32(
        svptrue_b16(),
        bases,
        U16_DATA.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u64base_index_s64() {
    let bases = svdup_n_u64(U16_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svsetffr();
    let _ = svld1uh_gather_u64base_index_s64(svptrue_b16(), bases, 1.try_into().unwrap());
    let loaded = svldff1uh_gather_u64base_index_s64(svptrue_b16(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_u64base_index_s64() {
    let bases = svdup_n_u64(U32_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svsetffr();
    let _ = svld1uw_gather_u64base_index_s64(svptrue_b32(), bases, 1.try_into().unwrap());
    let loaded = svldff1uw_gather_u64base_index_s64(svptrue_b32(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uh_gather_u64base_index_u64() {
    let bases = svdup_n_u64(U16_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svsetffr();
    let _ = svld1uh_gather_u64base_index_u64(svptrue_b16(), bases, 1.try_into().unwrap());
    let loaded = svldff1uh_gather_u64base_index_u64(svptrue_b16(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldff1uw_gather_u64base_index_u64() {
    let bases = svdup_n_u64(U32_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svsetffr();
    let _ = svld1uw_gather_u64base_index_u64(svptrue_b32(), bases, 1.try_into().unwrap());
    let loaded = svldff1uw_gather_u64base_index_u64(svptrue_b32(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_f32() {
    svsetffr();
    let _ = svld1_f32(svptrue_b32(), F32_DATA.as_ptr());
    let loaded = svldnf1_f32(svptrue_b32(), F32_DATA.as_ptr());
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_f64() {
    svsetffr();
    let _ = svld1_f64(svptrue_b64(), F64_DATA.as_ptr());
    let loaded = svldnf1_f64(svptrue_b64(), F64_DATA.as_ptr());
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_s8() {
    svsetffr();
    let _ = svld1_s8(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldnf1_s8(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_i8(
        loaded,
        svindex_s8((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_s16() {
    svsetffr();
    let _ = svld1_s16(svptrue_b16(), I16_DATA.as_ptr());
    let loaded = svldnf1_s16(svptrue_b16(), I16_DATA.as_ptr());
    assert_vector_matches_i16(
        loaded,
        svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_s32() {
    svsetffr();
    let _ = svld1_s32(svptrue_b32(), I32_DATA.as_ptr());
    let loaded = svldnf1_s32(svptrue_b32(), I32_DATA.as_ptr());
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_s64() {
    svsetffr();
    let _ = svld1_s64(svptrue_b64(), I64_DATA.as_ptr());
    let loaded = svldnf1_s64(svptrue_b64(), I64_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_u8() {
    svsetffr();
    let _ = svld1_u8(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldnf1_u8(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_u8(
        loaded,
        svindex_u8((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_u16() {
    svsetffr();
    let _ = svld1_u16(svptrue_b16(), U16_DATA.as_ptr());
    let loaded = svldnf1_u16(svptrue_b16(), U16_DATA.as_ptr());
    assert_vector_matches_u16(
        loaded,
        svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_u32() {
    svsetffr();
    let _ = svld1_u32(svptrue_b32(), U32_DATA.as_ptr());
    let loaded = svldnf1_u32(svptrue_b32(), U32_DATA.as_ptr());
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_u64() {
    svsetffr();
    let _ = svld1_u64(svptrue_b64(), U64_DATA.as_ptr());
    let loaded = svldnf1_u64(svptrue_b64(), U64_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_vnum_f32() {
    svsetffr();
    let _ = svld1_vnum_f32(svptrue_b32(), F32_DATA.as_ptr(), 1);
    let loaded = svldnf1_vnum_f32(svptrue_b32(), F32_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 0usize).try_into().unwrap(),
                1usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_vnum_f64() {
    svsetffr();
    let _ = svld1_vnum_f64(svptrue_b64(), F64_DATA.as_ptr(), 1);
    let loaded = svldnf1_vnum_f64(svptrue_b64(), F64_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 0usize).try_into().unwrap(),
                1usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_vnum_s8() {
    svsetffr();
    let _ = svld1_vnum_s8(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldnf1_vnum_s8(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcntb() as usize;
    assert_vector_matches_i8(
        loaded,
        svindex_s8(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_vnum_s16() {
    svsetffr();
    let _ = svld1_vnum_s16(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let loaded = svldnf1_vnum_s16(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let len = svcnth() as usize;
    assert_vector_matches_i16(
        loaded,
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_vnum_s32() {
    svsetffr();
    let _ = svld1_vnum_s32(svptrue_b32(), I32_DATA.as_ptr(), 1);
    let loaded = svldnf1_vnum_s32(svptrue_b32(), I32_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_vnum_s64() {
    svsetffr();
    let _ = svld1_vnum_s64(svptrue_b64(), I64_DATA.as_ptr(), 1);
    let loaded = svldnf1_vnum_s64(svptrue_b64(), I64_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_vnum_u8() {
    svsetffr();
    let _ = svld1_vnum_u8(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldnf1_vnum_u8(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcntb() as usize;
    assert_vector_matches_u8(
        loaded,
        svindex_u8(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_vnum_u16() {
    svsetffr();
    let _ = svld1_vnum_u16(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let loaded = svldnf1_vnum_u16(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let len = svcnth() as usize;
    assert_vector_matches_u16(
        loaded,
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_vnum_u32() {
    svsetffr();
    let _ = svld1_vnum_u32(svptrue_b32(), U32_DATA.as_ptr(), 1);
    let loaded = svldnf1_vnum_u32(svptrue_b32(), U32_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1_vnum_u64() {
    svsetffr();
    let _ = svld1_vnum_u64(svptrue_b64(), U64_DATA.as_ptr(), 1);
    let loaded = svldnf1_vnum_u64(svptrue_b64(), U64_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sb_s16() {
    svsetffr();
    let _ = svld1sb_s16(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldnf1sb_s16(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_i16(
        loaded,
        svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sb_s32() {
    svsetffr();
    let _ = svld1sb_s32(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldnf1sb_s32(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sh_s32() {
    svsetffr();
    let _ = svld1sh_s32(svptrue_b16(), I16_DATA.as_ptr());
    let loaded = svldnf1sh_s32(svptrue_b16(), I16_DATA.as_ptr());
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sb_s64() {
    svsetffr();
    let _ = svld1sb_s64(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldnf1sb_s64(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sh_s64() {
    svsetffr();
    let _ = svld1sh_s64(svptrue_b16(), I16_DATA.as_ptr());
    let loaded = svldnf1sh_s64(svptrue_b16(), I16_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sw_s64() {
    svsetffr();
    let _ = svld1sw_s64(svptrue_b32(), I32_DATA.as_ptr());
    let loaded = svldnf1sw_s64(svptrue_b32(), I32_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sb_u16() {
    svsetffr();
    let _ = svld1sb_u16(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldnf1sb_u16(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_u16(
        loaded,
        svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sb_u32() {
    svsetffr();
    let _ = svld1sb_u32(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldnf1sb_u32(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sh_u32() {
    svsetffr();
    let _ = svld1sh_u32(svptrue_b16(), I16_DATA.as_ptr());
    let loaded = svldnf1sh_u32(svptrue_b16(), I16_DATA.as_ptr());
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sb_u64() {
    svsetffr();
    let _ = svld1sb_u64(svptrue_b8(), I8_DATA.as_ptr());
    let loaded = svldnf1sb_u64(svptrue_b8(), I8_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sh_u64() {
    svsetffr();
    let _ = svld1sh_u64(svptrue_b16(), I16_DATA.as_ptr());
    let loaded = svldnf1sh_u64(svptrue_b16(), I16_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sw_u64() {
    svsetffr();
    let _ = svld1sw_u64(svptrue_b32(), I32_DATA.as_ptr());
    let loaded = svldnf1sw_u64(svptrue_b32(), I32_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sb_vnum_s16() {
    svsetffr();
    let _ = svld1sb_vnum_s16(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldnf1sb_vnum_s16(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcnth() as usize;
    assert_vector_matches_i16(
        loaded,
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sb_vnum_s32() {
    svsetffr();
    let _ = svld1sb_vnum_s32(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldnf1sb_vnum_s32(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sh_vnum_s32() {
    svsetffr();
    let _ = svld1sh_vnum_s32(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let loaded = svldnf1sh_vnum_s32(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sb_vnum_s64() {
    svsetffr();
    let _ = svld1sb_vnum_s64(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldnf1sb_vnum_s64(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sh_vnum_s64() {
    svsetffr();
    let _ = svld1sh_vnum_s64(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let loaded = svldnf1sh_vnum_s64(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sw_vnum_s64() {
    svsetffr();
    let _ = svld1sw_vnum_s64(svptrue_b32(), I32_DATA.as_ptr(), 1);
    let loaded = svldnf1sw_vnum_s64(svptrue_b32(), I32_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sb_vnum_u16() {
    svsetffr();
    let _ = svld1sb_vnum_u16(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldnf1sb_vnum_u16(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcnth() as usize;
    assert_vector_matches_u16(
        loaded,
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sb_vnum_u32() {
    svsetffr();
    let _ = svld1sb_vnum_u32(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldnf1sb_vnum_u32(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sh_vnum_u32() {
    svsetffr();
    let _ = svld1sh_vnum_u32(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let loaded = svldnf1sh_vnum_u32(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sb_vnum_u64() {
    svsetffr();
    let _ = svld1sb_vnum_u64(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let loaded = svldnf1sb_vnum_u64(svptrue_b8(), I8_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sh_vnum_u64() {
    svsetffr();
    let _ = svld1sh_vnum_u64(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let loaded = svldnf1sh_vnum_u64(svptrue_b16(), I16_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1sw_vnum_u64() {
    svsetffr();
    let _ = svld1sw_vnum_u64(svptrue_b32(), I32_DATA.as_ptr(), 1);
    let loaded = svldnf1sw_vnum_u64(svptrue_b32(), I32_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1ub_s16() {
    svsetffr();
    let _ = svld1ub_s16(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldnf1ub_s16(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_i16(
        loaded,
        svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1ub_s32() {
    svsetffr();
    let _ = svld1ub_s32(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldnf1ub_s32(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1uh_s32() {
    svsetffr();
    let _ = svld1uh_s32(svptrue_b16(), U16_DATA.as_ptr());
    let loaded = svldnf1uh_s32(svptrue_b16(), U16_DATA.as_ptr());
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1ub_s64() {
    svsetffr();
    let _ = svld1ub_s64(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldnf1ub_s64(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1uh_s64() {
    svsetffr();
    let _ = svld1uh_s64(svptrue_b16(), U16_DATA.as_ptr());
    let loaded = svldnf1uh_s64(svptrue_b16(), U16_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1uw_s64() {
    svsetffr();
    let _ = svld1uw_s64(svptrue_b32(), U32_DATA.as_ptr());
    let loaded = svldnf1uw_s64(svptrue_b32(), U32_DATA.as_ptr());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1ub_u16() {
    svsetffr();
    let _ = svld1ub_u16(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldnf1ub_u16(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_u16(
        loaded,
        svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1ub_u32() {
    svsetffr();
    let _ = svld1ub_u32(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldnf1ub_u32(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1uh_u32() {
    svsetffr();
    let _ = svld1uh_u32(svptrue_b16(), U16_DATA.as_ptr());
    let loaded = svldnf1uh_u32(svptrue_b16(), U16_DATA.as_ptr());
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1ub_u64() {
    svsetffr();
    let _ = svld1ub_u64(svptrue_b8(), U8_DATA.as_ptr());
    let loaded = svldnf1ub_u64(svptrue_b8(), U8_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1uh_u64() {
    svsetffr();
    let _ = svld1uh_u64(svptrue_b16(), U16_DATA.as_ptr());
    let loaded = svldnf1uh_u64(svptrue_b16(), U16_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1uw_u64() {
    svsetffr();
    let _ = svld1uw_u64(svptrue_b32(), U32_DATA.as_ptr());
    let loaded = svldnf1uw_u64(svptrue_b32(), U32_DATA.as_ptr());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1ub_vnum_s16() {
    svsetffr();
    let _ = svld1ub_vnum_s16(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldnf1ub_vnum_s16(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcnth() as usize;
    assert_vector_matches_i16(
        loaded,
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1ub_vnum_s32() {
    svsetffr();
    let _ = svld1ub_vnum_s32(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldnf1ub_vnum_s32(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1uh_vnum_s32() {
    svsetffr();
    let _ = svld1uh_vnum_s32(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let loaded = svldnf1uh_vnum_s32(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1ub_vnum_s64() {
    svsetffr();
    let _ = svld1ub_vnum_s64(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldnf1ub_vnum_s64(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1uh_vnum_s64() {
    svsetffr();
    let _ = svld1uh_vnum_s64(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let loaded = svldnf1uh_vnum_s64(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1uw_vnum_s64() {
    svsetffr();
    let _ = svld1uw_vnum_s64(svptrue_b32(), U32_DATA.as_ptr(), 1);
    let loaded = svldnf1uw_vnum_s64(svptrue_b32(), U32_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1ub_vnum_u16() {
    svsetffr();
    let _ = svld1ub_vnum_u16(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldnf1ub_vnum_u16(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcnth() as usize;
    assert_vector_matches_u16(
        loaded,
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1ub_vnum_u32() {
    svsetffr();
    let _ = svld1ub_vnum_u32(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldnf1ub_vnum_u32(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1uh_vnum_u32() {
    svsetffr();
    let _ = svld1uh_vnum_u32(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let loaded = svldnf1uh_vnum_u32(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let len = svcntw() as usize;
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1ub_vnum_u64() {
    svsetffr();
    let _ = svld1ub_vnum_u64(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let loaded = svldnf1ub_vnum_u64(svptrue_b8(), U8_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1uh_vnum_u64() {
    svsetffr();
    let _ = svld1uh_vnum_u64(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let loaded = svldnf1uh_vnum_u64(svptrue_b16(), U16_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnf1uw_vnum_u64() {
    svsetffr();
    let _ = svld1uw_vnum_u64(svptrue_b32(), U32_DATA.as_ptr(), 1);
    let loaded = svldnf1uw_vnum_u64(svptrue_b32(), U32_DATA.as_ptr(), 1);
    let len = svcntd() as usize;
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_f32_with_svstnt1_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    svstnt1_f32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svldnt1_f32(svptrue_b32(), storage.as_ptr() as *const f32);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_f64_with_svstnt1_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    svstnt1_f64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svldnt1_f64(svptrue_b64(), storage.as_ptr() as *const f64);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_s8_with_svstnt1_s8() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s8((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svstnt1_s8(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1_s8(svptrue_b8(), storage.as_ptr() as *const i8);
    assert_vector_matches_i8(
        loaded,
        svindex_s8((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_s16_with_svstnt1_s16() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svstnt1_s16(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1_s16(svptrue_b16(), storage.as_ptr() as *const i16);
    assert_vector_matches_i16(
        loaded,
        svindex_s16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_s32_with_svstnt1_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svstnt1_s32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svldnt1_s32(svptrue_b32(), storage.as_ptr() as *const i32);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_s64_with_svstnt1_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svstnt1_s64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svldnt1_s64(svptrue_b64(), storage.as_ptr() as *const i64);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_u8_with_svstnt1_u8() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u8((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svstnt1_u8(svptrue_b8(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svldnt1_u8(svptrue_b8(), storage.as_ptr() as *const u8);
    assert_vector_matches_u8(
        loaded,
        svindex_u8((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_u16_with_svstnt1_u16() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svstnt1_u16(svptrue_b16(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svldnt1_u16(svptrue_b16(), storage.as_ptr() as *const u16);
    assert_vector_matches_u16(
        loaded,
        svindex_u16((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_u32_with_svstnt1_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svstnt1_u32(svptrue_b32(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svldnt1_u32(svptrue_b32(), storage.as_ptr() as *const u32);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_u64_with_svstnt1_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    svstnt1_u64(svptrue_b64(), storage.as_mut_ptr(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svldnt1_u64(svptrue_b64(), storage.as_ptr() as *const u64);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_vnum_f32_with_svstnt1_vnum_f32() {
    let len = svcntw() as usize;
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
    svstnt1_vnum_f32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svldnt1_vnum_f32(svptrue_b32(), storage.as_ptr() as *const f32, 1);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32(
                (len + 0usize).try_into().unwrap(),
                1usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_vnum_f64_with_svstnt1_vnum_f64() {
    let len = svcntd() as usize;
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
    svstnt1_vnum_f64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svldnt1_vnum_f64(svptrue_b64(), storage.as_ptr() as *const f64, 1);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64(
                (len + 0usize).try_into().unwrap(),
                1usize.try_into().unwrap(),
            ),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_vnum_s8_with_svstnt1_vnum_s8() {
    let len = svcntb() as usize;
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s8(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svstnt1_vnum_s8(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1_vnum_s8(svptrue_b8(), storage.as_ptr() as *const i8, 1);
    assert_vector_matches_i8(
        loaded,
        svindex_s8(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_vnum_s16_with_svstnt1_vnum_s16() {
    let len = svcnth() as usize;
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s16(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svstnt1_vnum_s16(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1_vnum_s16(svptrue_b16(), storage.as_ptr() as *const i16, 1);
    assert_vector_matches_i16(
        loaded,
        svindex_s16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_vnum_s32_with_svstnt1_vnum_s32() {
    let len = svcntw() as usize;
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svstnt1_vnum_s32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svldnt1_vnum_s32(svptrue_b32(), storage.as_ptr() as *const i32, 1);
    assert_vector_matches_i32(
        loaded,
        svindex_s32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_vnum_s64_with_svstnt1_vnum_s64() {
    let len = svcntd() as usize;
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svstnt1_vnum_s64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svldnt1_vnum_s64(svptrue_b64(), storage.as_ptr() as *const i64, 1);
    assert_vector_matches_i64(
        loaded,
        svindex_s64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_vnum_u8_with_svstnt1_vnum_u8() {
    let len = svcntb() as usize;
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u8(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svstnt1_vnum_u8(svptrue_b8(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded = svldnt1_vnum_u8(svptrue_b8(), storage.as_ptr() as *const u8, 1);
    assert_vector_matches_u8(
        loaded,
        svindex_u8(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_vnum_u16_with_svstnt1_vnum_u16() {
    let len = svcnth() as usize;
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u16(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svstnt1_vnum_u16(svptrue_b16(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded = svldnt1_vnum_u16(svptrue_b16(), storage.as_ptr() as *const u16, 1);
    assert_vector_matches_u16(
        loaded,
        svindex_u16(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_vnum_u32_with_svstnt1_vnum_u32() {
    let len = svcntw() as usize;
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svstnt1_vnum_u32(svptrue_b32(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svldnt1_vnum_u32(svptrue_b32(), storage.as_ptr() as *const u32, 1);
    assert_vector_matches_u32(
        loaded,
        svindex_u32(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svldnt1_vnum_u64_with_svstnt1_vnum_u64() {
    let len = svcntd() as usize;
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64(
        (len + 0usize).try_into().unwrap(),
        1usize.try_into().unwrap(),
    );
    svstnt1_vnum_u64(svptrue_b64(), storage.as_mut_ptr(), 1, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svldnt1_vnum_u64(svptrue_b64(), storage.as_ptr() as *const u64, 1);
    assert_vector_matches_u64(
        loaded,
        svindex_u64(
            (len + 0usize).try_into().unwrap(),
            1usize.try_into().unwrap(),
        ),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfb() {
    svsetffr();
    let loaded = svprfb::<{ svprfop::SV_PLDL1KEEP }, i64>(svptrue_b8(), I64_DATA.as_ptr());
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfh() {
    svsetffr();
    let loaded = svprfh::<{ svprfop::SV_PLDL1KEEP }, i64>(svptrue_b16(), I64_DATA.as_ptr());
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfw() {
    svsetffr();
    let loaded = svprfw::<{ svprfop::SV_PLDL1KEEP }, i64>(svptrue_b32(), I64_DATA.as_ptr());
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfd() {
    svsetffr();
    let loaded = svprfd::<{ svprfop::SV_PLDL1KEEP }, i64>(svptrue_b64(), I64_DATA.as_ptr());
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfb_gather_s32offset() {
    let offsets = svindex_s32(0, 4u32.try_into().unwrap());
    svsetffr();
    let loaded = svprfb_gather_s32offset::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b32(),
        I64_DATA.as_ptr(),
        offsets,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfh_gather_s32index() {
    let indices = svindex_s32(0, 1);
    svsetffr();
    let loaded = svprfh_gather_s32index::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b32(),
        I64_DATA.as_ptr(),
        indices,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfw_gather_s32index() {
    let indices = svindex_s32(0, 1);
    svsetffr();
    let loaded = svprfw_gather_s32index::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b32(),
        I64_DATA.as_ptr(),
        indices,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfd_gather_s32index() {
    let indices = svindex_s32(0, 1);
    svsetffr();
    let loaded = svprfd_gather_s32index::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b32(),
        I64_DATA.as_ptr(),
        indices,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfb_gather_s64offset() {
    let offsets = svindex_s64(0, 8u32.try_into().unwrap());
    svsetffr();
    let loaded = svprfb_gather_s64offset::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b64(),
        I64_DATA.as_ptr(),
        offsets,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfh_gather_s64index() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let loaded = svprfh_gather_s64index::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b64(),
        I64_DATA.as_ptr(),
        indices,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfw_gather_s64index() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let loaded = svprfw_gather_s64index::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b64(),
        I64_DATA.as_ptr(),
        indices,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfd_gather_s64index() {
    let indices = svindex_s64(0, 1);
    svsetffr();
    let loaded = svprfd_gather_s64index::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b64(),
        I64_DATA.as_ptr(),
        indices,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfb_gather_u32offset() {
    let offsets = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let loaded = svprfb_gather_u32offset::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b32(),
        I64_DATA.as_ptr(),
        offsets,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfh_gather_u32index() {
    let indices = svindex_u32(0, 1);
    svsetffr();
    let loaded = svprfh_gather_u32index::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b32(),
        I64_DATA.as_ptr(),
        indices,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfw_gather_u32index() {
    let indices = svindex_u32(0, 1);
    svsetffr();
    let loaded = svprfw_gather_u32index::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b32(),
        I64_DATA.as_ptr(),
        indices,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfd_gather_u32index() {
    let indices = svindex_u32(0, 1);
    svsetffr();
    let loaded = svprfd_gather_u32index::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b32(),
        I64_DATA.as_ptr(),
        indices,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfb_gather_u64offset() {
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    svsetffr();
    let loaded = svprfb_gather_u64offset::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b64(),
        I64_DATA.as_ptr(),
        offsets,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfh_gather_u64index() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let loaded = svprfh_gather_u64index::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b64(),
        I64_DATA.as_ptr(),
        indices,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfw_gather_u64index() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let loaded = svprfw_gather_u64index::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b64(),
        I64_DATA.as_ptr(),
        indices,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfd_gather_u64index() {
    let indices = svindex_u64(0, 1);
    svsetffr();
    let loaded = svprfd_gather_u64index::<{ svprfop::SV_PLDL1KEEP }, i64>(
        svptrue_b64(),
        I64_DATA.as_ptr(),
        indices,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfb_gather_u64base() {
    let bases = svdup_n_u64(U64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let loaded = svprfb_gather_u64base::<{ svprfop::SV_PLDL1KEEP }>(svptrue_b64(), bases);
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfh_gather_u64base() {
    let bases = svdup_n_u64(U64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let loaded = svprfh_gather_u64base::<{ svprfop::SV_PLDL1KEEP }>(svptrue_b64(), bases);
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfw_gather_u64base() {
    let bases = svdup_n_u64(U64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let loaded = svprfw_gather_u64base::<{ svprfop::SV_PLDL1KEEP }>(svptrue_b64(), bases);
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfd_gather_u64base() {
    let bases = svdup_n_u64(U64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let loaded = svprfd_gather_u64base::<{ svprfop::SV_PLDL1KEEP }>(svptrue_b64(), bases);
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfb_gather_u32base_offset() {
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let loaded = svprfb_gather_u32base_offset::<{ svprfop::SV_PLDL1KEEP }>(
        svptrue_b32(),
        bases,
        U32_DATA.as_ptr() as i64 + 4u32 as i64,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfh_gather_u32base_index() {
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let loaded = svprfh_gather_u32base_index::<{ svprfop::SV_PLDL1KEEP }>(
        svptrue_b32(),
        bases,
        U32_DATA.as_ptr() as i64 / (4u32 as i64) + 1,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfw_gather_u32base_index() {
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let loaded = svprfw_gather_u32base_index::<{ svprfop::SV_PLDL1KEEP }>(
        svptrue_b32(),
        bases,
        U32_DATA.as_ptr() as i64 / (4u32 as i64) + 1,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfd_gather_u32base_index() {
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svsetffr();
    let loaded = svprfd_gather_u32base_index::<{ svprfop::SV_PLDL1KEEP }>(
        svptrue_b32(),
        bases,
        U32_DATA.as_ptr() as i64 / (4u32 as i64) + 1,
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfb_gather_u64base_offset() {
    let bases = svdup_n_u64(U64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let loaded = svprfb_gather_u64base_offset::<{ svprfop::SV_PLDL1KEEP }>(
        svptrue_b64(),
        bases,
        8u32.try_into().unwrap(),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfh_gather_u64base_index() {
    let bases = svdup_n_u64(U64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let loaded = svprfh_gather_u64base_index::<{ svprfop::SV_PLDL1KEEP }>(
        svptrue_b64(),
        bases,
        1.try_into().unwrap(),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfw_gather_u64base_index() {
    let bases = svdup_n_u64(U64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let loaded = svprfw_gather_u64base_index::<{ svprfop::SV_PLDL1KEEP }>(
        svptrue_b64(),
        bases,
        1.try_into().unwrap(),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfd_gather_u64base_index() {
    let bases = svdup_n_u64(U64_DATA.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svsetffr();
    let loaded = svprfd_gather_u64base_index::<{ svprfop::SV_PLDL1KEEP }>(
        svptrue_b64(),
        bases,
        1.try_into().unwrap(),
    );
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfb_vnum() {
    svsetffr();
    let loaded = svprfb_vnum::<{ svprfop::SV_PLDL1KEEP }, i64>(svptrue_b8(), I64_DATA.as_ptr(), 1);
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfh_vnum() {
    svsetffr();
    let loaded = svprfh_vnum::<{ svprfop::SV_PLDL1KEEP }, i64>(svptrue_b16(), I64_DATA.as_ptr(), 1);
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfw_vnum() {
    svsetffr();
    let loaded = svprfw_vnum::<{ svprfop::SV_PLDL1KEEP }, i64>(svptrue_b32(), I64_DATA.as_ptr(), 1);
}
#[simd_test(enable = "sve")]
unsafe fn test_svprfd_vnum() {
    svsetffr();
    let loaded = svprfd_vnum::<{ svprfop::SV_PLDL1KEEP }, i64>(svptrue_b64(), I64_DATA.as_ptr(), 1);
}
#[simd_test(enable = "sve")]
unsafe fn test_ffr() {
    svsetffr();
    let ffr = svrdffr();
    assert_vector_matches_u8(svdup_n_u8_z(ffr, 1), svindex_u8(1, 0));
    let pred = svdupq_n_b8(
        true, false, true, false, true, false, true, false, true, false, true, false, true, false,
        true, false,
    );
    svwrffr(pred);
    let ffr = svrdffr_z(svptrue_b8());
    assert_vector_matches_u8(svdup_n_u8_z(ffr, 1), svdup_n_u8_z(pred, 1));
}
