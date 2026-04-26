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
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_s64index_f64_with_svstnt1_scatter_s64index_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let indices = svindex_s64(0, 1);
    svstnt1_scatter_s64index_f64(svptrue_b64(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_s64index_f64(svptrue_b64(), storage.as_ptr() as *const f64, indices);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_s64index_s64_with_svstnt1_scatter_s64index_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svstnt1_scatter_s64index_s64(svptrue_b64(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_s64index_s64(svptrue_b64(), storage.as_ptr() as *const i64, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_s64index_u64_with_svstnt1_scatter_s64index_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svstnt1_scatter_s64index_u64(svptrue_b64(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_s64index_u64(svptrue_b64(), storage.as_ptr() as *const u64, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64index_f64_with_svstnt1_scatter_u64index_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let indices = svindex_u64(0, 1);
    svstnt1_scatter_u64index_f64(svptrue_b64(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_u64index_f64(svptrue_b64(), storage.as_ptr() as *const f64, indices);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64index_s64_with_svstnt1_scatter_u64index_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svstnt1_scatter_u64index_s64(svptrue_b64(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_u64index_s64(svptrue_b64(), storage.as_ptr() as *const i64, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64index_u64_with_svstnt1_scatter_u64index_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svstnt1_scatter_u64index_u64(svptrue_b64(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_u64index_u64(svptrue_b64(), storage.as_ptr() as *const u64, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_s64offset_f64_with_svstnt1_scatter_s64offset_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let offsets = svindex_s64(0, 8u32.try_into().unwrap());
    svstnt1_scatter_s64offset_f64(svptrue_b64(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_s64offset_f64(svptrue_b64(), storage.as_ptr() as *const f64, offsets);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_s64offset_s64_with_svstnt1_scatter_s64offset_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 8u32.try_into().unwrap());
    svstnt1_scatter_s64offset_s64(svptrue_b64(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_s64offset_s64(svptrue_b64(), storage.as_ptr() as *const i64, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_s64offset_u64_with_svstnt1_scatter_s64offset_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 8u32.try_into().unwrap());
    svstnt1_scatter_s64offset_u64(svptrue_b64(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_s64offset_u64(svptrue_b64(), storage.as_ptr() as *const u64, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u32offset_f32_with_svstnt1_scatter_u32offset_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let offsets = svindex_u32(0, 4u32.try_into().unwrap());
    svstnt1_scatter_u32offset_f32(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_u32offset_f32(svptrue_b32(), storage.as_ptr() as *const f32, offsets);
    assert_vector_matches_f32(
        loaded,
        svcvt_f32_s32_x(
            svptrue_b32(),
            svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u32offset_s32_with_svstnt1_scatter_u32offset_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 4u32.try_into().unwrap());
    svstnt1_scatter_u32offset_s32(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_u32offset_s32(svptrue_b32(), storage.as_ptr() as *const i32, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u32offset_u32_with_svstnt1_scatter_u32offset_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 4u32.try_into().unwrap());
    svstnt1_scatter_u32offset_u32(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_u32offset_u32(svptrue_b32(), storage.as_ptr() as *const u32, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64offset_f64_with_svstnt1_scatter_u64offset_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    svstnt1_scatter_u64offset_f64(svptrue_b64(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_u64offset_f64(svptrue_b64(), storage.as_ptr() as *const f64, offsets);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64offset_s64_with_svstnt1_scatter_u64offset_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    svstnt1_scatter_u64offset_s64(svptrue_b64(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_u64offset_s64(svptrue_b64(), storage.as_ptr() as *const i64, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64offset_u64_with_svstnt1_scatter_u64offset_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    svstnt1_scatter_u64offset_u64(svptrue_b64(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded =
        svldnt1_gather_u64offset_u64(svptrue_b64(), storage.as_ptr() as *const u64, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64base_f64_with_svstnt1_scatter_u64base_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svstnt1_scatter_u64base_f64(svptrue_b64(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svldnt1_gather_u64base_f64(svptrue_b64(), bases);
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64base_s64_with_svstnt1_scatter_u64base_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svstnt1_scatter_u64base_s64(svptrue_b64(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svldnt1_gather_u64base_s64(svptrue_b64(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64base_u64_with_svstnt1_scatter_u64base_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svstnt1_scatter_u64base_u64(svptrue_b64(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svldnt1_gather_u64base_u64(svptrue_b64(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u32base_index_f32_with_svstnt1_scatter_u32base_index_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svstnt1_scatter_u32base_index_f32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 / (4u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svldnt1_gather_u32base_index_f32(
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
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u32base_index_s32_with_svstnt1_scatter_u32base_index_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svstnt1_scatter_u32base_index_s32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 / (4u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svldnt1_gather_u32base_index_s32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 / (4u32 as i64) + 1,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u32base_index_u32_with_svstnt1_scatter_u32base_index_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svstnt1_scatter_u32base_index_u32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 / (4u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svldnt1_gather_u32base_index_u32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 / (4u32 as i64) + 1,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64base_index_f64_with_svstnt1_scatter_u64base_index_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svstnt1_scatter_u64base_index_f64(svptrue_b64(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svldnt1_gather_u64base_index_f64(svptrue_b64(), bases, 1.try_into().unwrap());
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64base_index_s64_with_svstnt1_scatter_u64base_index_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svstnt1_scatter_u64base_index_s64(svptrue_b64(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svldnt1_gather_u64base_index_s64(svptrue_b64(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64base_index_u64_with_svstnt1_scatter_u64base_index_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svstnt1_scatter_u64base_index_u64(svptrue_b64(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svldnt1_gather_u64base_index_u64(svptrue_b64(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u32base_offset_f32_with_svstnt1_scatter_u32base_offset_f32() {
    let mut storage = [0 as f32; 320usize];
    let data = svcvt_f32_s32_x(
        svptrue_b32(),
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svstnt1_scatter_u32base_offset_f32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 + 4u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f32 || val == i as f32);
    }
    svsetffr();
    let loaded = svldnt1_gather_u32base_offset_f32(
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
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u32base_offset_s32_with_svstnt1_scatter_u32base_offset_s32() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svstnt1_scatter_u32base_offset_s32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 + 4u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svldnt1_gather_u32base_offset_s32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 + 4u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u32base_offset_u32_with_svstnt1_scatter_u32base_offset_u32() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 4u32.try_into().unwrap());
    svstnt1_scatter_u32base_offset_u32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 + 4u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded = svldnt1_gather_u32base_offset_u32(
        svptrue_b32(),
        bases,
        storage.as_ptr() as i64 + 4u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64base_offset_f64_with_svstnt1_scatter_u64base_offset_f64() {
    let mut storage = [0 as f64; 160usize];
    let data = svcvt_f64_s64_x(
        svptrue_b64(),
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svstnt1_scatter_u64base_offset_f64(svptrue_b64(), bases, 8u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as f64 || val == i as f64);
    }
    svsetffr();
    let loaded = svldnt1_gather_u64base_offset_f64(svptrue_b64(), bases, 8u32.try_into().unwrap());
    assert_vector_matches_f64(
        loaded,
        svcvt_f64_s64_x(
            svptrue_b64(),
            svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
        ),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64base_offset_s64_with_svstnt1_scatter_u64base_offset_s64() {
    let mut storage = [0 as i64; 160usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svstnt1_scatter_u64base_offset_s64(svptrue_b64(), bases, 8u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i64 || val == i as i64);
    }
    svsetffr();
    let loaded = svldnt1_gather_u64base_offset_s64(svptrue_b64(), bases, 8u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1_gather_u64base_offset_u64_with_svstnt1_scatter_u64base_offset_u64() {
    let mut storage = [0 as u64; 160usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 8u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b64(), bases, offsets);
    svstnt1_scatter_u64base_offset_u64(svptrue_b64(), bases, 8u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u64 || val == i as u64);
    }
    svsetffr();
    let loaded = svldnt1_gather_u64base_offset_u64(svptrue_b64(), bases, 8u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sb_gather_s64offset_s64_with_svstnt1b_scatter_s64offset_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_s64offset_s64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded =
        svldnt1sb_gather_s64offset_s64(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_s64offset_s64_with_svstnt1h_scatter_s64offset_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_s64offset_s64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1sh_gather_s64offset_s64(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_s64offset_s64_with_svstnt1w_scatter_s64offset_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 4u32.try_into().unwrap());
    svstnt1w_scatter_s64offset_s64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1sw_gather_s64offset_s64(svptrue_b32(), storage.as_ptr() as *const i32, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sb_gather_s64offset_u64_with_svstnt1b_scatter_s64offset_u64() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_s64offset_u64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded =
        svldnt1sb_gather_s64offset_u64(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_s64offset_u64_with_svstnt1h_scatter_s64offset_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_s64offset_u64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svldnt1sh_gather_s64offset_u64(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_s64offset_u64_with_svstnt1w_scatter_s64offset_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 4u32.try_into().unwrap());
    svstnt1w_scatter_s64offset_u64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svldnt1sw_gather_s64offset_u64(svptrue_b32(), storage.as_ptr() as *const i32, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sb_gather_u32offset_s32_with_svstnt1b_scatter_u32offset_s32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_u32offset_s32(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded =
        svldnt1sb_gather_u32offset_s32(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u32offset_s32_with_svstnt1h_scatter_u32offset_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u32offset_s32(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1sh_gather_u32offset_s32(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sb_gather_u32offset_u32_with_svstnt1b_scatter_u32offset_u32() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_u32offset_u32(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded =
        svldnt1sb_gather_u32offset_u32(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u32offset_u32_with_svstnt1h_scatter_u32offset_u32() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u32offset_u32(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svldnt1sh_gather_u32offset_u32(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sb_gather_u64offset_s64_with_svstnt1b_scatter_u64offset_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_u64offset_s64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded =
        svldnt1sb_gather_u64offset_s64(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u64offset_s64_with_svstnt1h_scatter_u64offset_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u64offset_s64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1sh_gather_u64offset_s64(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_u64offset_s64_with_svstnt1w_scatter_u64offset_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    svstnt1w_scatter_u64offset_s64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1sw_gather_u64offset_s64(svptrue_b32(), storage.as_ptr() as *const i32, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sb_gather_u64offset_u64_with_svstnt1b_scatter_u64offset_u64() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_u64offset_u64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded =
        svldnt1sb_gather_u64offset_u64(svptrue_b8(), storage.as_ptr() as *const i8, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u64offset_u64_with_svstnt1h_scatter_u64offset_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u64offset_u64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svldnt1sh_gather_u64offset_u64(svptrue_b16(), storage.as_ptr() as *const i16, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_u64offset_u64_with_svstnt1w_scatter_u64offset_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    svstnt1w_scatter_u64offset_u64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svldnt1sw_gather_u64offset_u64(svptrue_b32(), storage.as_ptr() as *const i32, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sb_gather_u32base_offset_s32_with_svstnt1b_scatter_u32base_offset_s32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_u32base_offset_s32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1sb_gather_u32base_offset_s32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u32base_offset_s32_with_svstnt1h_scatter_u32base_offset_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u32base_offset_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1sh_gather_u32base_offset_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sb_gather_u32base_offset_u32_with_svstnt1b_scatter_u32base_offset_u32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_u32base_offset_u32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1sb_gather_u32base_offset_u32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u32base_offset_u32_with_svstnt1h_scatter_u32base_offset_u32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u32base_offset_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1sh_gather_u32base_offset_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sb_gather_u64base_offset_s64_with_svstnt1b_scatter_u64base_offset_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svstnt1b_scatter_u64base_offset_s64(svptrue_b8(), bases, 1u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1sb_gather_u64base_offset_s64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u64base_offset_s64_with_svstnt1h_scatter_u64base_offset_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svstnt1h_scatter_u64base_offset_s64(svptrue_b16(), bases, 2u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1sh_gather_u64base_offset_s64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_u64base_offset_s64_with_svstnt1w_scatter_u64base_offset_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svstnt1w_scatter_u64base_offset_s64(svptrue_b32(), bases, 4u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1sw_gather_u64base_offset_s64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sb_gather_u64base_offset_u64_with_svstnt1b_scatter_u64base_offset_u64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svstnt1b_scatter_u64base_offset_u64(svptrue_b8(), bases, 1u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1sb_gather_u64base_offset_u64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u64base_offset_u64_with_svstnt1h_scatter_u64base_offset_u64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svstnt1h_scatter_u64base_offset_u64(svptrue_b16(), bases, 2u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1sh_gather_u64base_offset_u64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_u64base_offset_u64_with_svstnt1w_scatter_u64base_offset_u64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svstnt1w_scatter_u64base_offset_u64(svptrue_b32(), bases, 4u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1sw_gather_u64base_offset_u64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sb_gather_u64base_s64_with_svstnt1b_scatter_u64base_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svstnt1b_scatter_u64base_s64(svptrue_b8(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1sb_gather_u64base_s64(svptrue_b8(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u64base_s64_with_svstnt1h_scatter_u64base_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svstnt1h_scatter_u64base_s64(svptrue_b16(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1sh_gather_u64base_s64(svptrue_b16(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_u64base_s64_with_svstnt1w_scatter_u64base_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svstnt1w_scatter_u64base_s64(svptrue_b32(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svldnt1sw_gather_u64base_s64(svptrue_b32(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sb_gather_u64base_u64_with_svstnt1b_scatter_u64base_u64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svstnt1b_scatter_u64base_u64(svptrue_b8(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1sb_gather_u64base_u64(svptrue_b8(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u64base_u64_with_svstnt1h_scatter_u64base_u64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svstnt1h_scatter_u64base_u64(svptrue_b16(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1sh_gather_u64base_u64(svptrue_b16(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_u64base_u64_with_svstnt1w_scatter_u64base_u64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svstnt1w_scatter_u64base_u64(svptrue_b32(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svldnt1sw_gather_u64base_u64(svptrue_b32(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_s64index_s64_with_svstnt1h_scatter_s64index_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svstnt1h_scatter_s64index_s64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1sh_gather_s64index_s64(svptrue_b16(), storage.as_ptr() as *const i16, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_s64index_s64_with_svstnt1w_scatter_s64index_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svstnt1w_scatter_s64index_s64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1sw_gather_s64index_s64(svptrue_b32(), storage.as_ptr() as *const i32, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_s64index_u64_with_svstnt1h_scatter_s64index_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svstnt1h_scatter_s64index_u64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svldnt1sh_gather_s64index_u64(svptrue_b16(), storage.as_ptr() as *const i16, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_s64index_u64_with_svstnt1w_scatter_s64index_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svstnt1w_scatter_s64index_u64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svldnt1sw_gather_s64index_u64(svptrue_b32(), storage.as_ptr() as *const i32, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u64index_s64_with_svstnt1h_scatter_u64index_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svstnt1h_scatter_u64index_s64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1sh_gather_u64index_s64(svptrue_b16(), storage.as_ptr() as *const i16, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_u64index_s64_with_svstnt1w_scatter_u64index_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svstnt1w_scatter_u64index_s64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1sw_gather_u64index_s64(svptrue_b32(), storage.as_ptr() as *const i32, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u64index_u64_with_svstnt1h_scatter_u64index_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svstnt1h_scatter_u64index_u64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svldnt1sh_gather_u64index_u64(svptrue_b16(), storage.as_ptr() as *const i16, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_u64index_u64_with_svstnt1w_scatter_u64index_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svstnt1w_scatter_u64index_u64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svldnt1sw_gather_u64index_u64(svptrue_b32(), storage.as_ptr() as *const i32, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u32base_index_s32_with_svstnt1h_scatter_u32base_index_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u32base_index_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1sh_gather_u32base_index_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u32base_index_u32_with_svstnt1h_scatter_u32base_index_u32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u32base_index_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1sh_gather_u32base_index_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u64base_index_s64_with_svstnt1h_scatter_u64base_index_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svstnt1h_scatter_u64base_index_s64(svptrue_b16(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1sh_gather_u64base_index_s64(svptrue_b16(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_u64base_index_s64_with_svstnt1w_scatter_u64base_index_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svstnt1w_scatter_u64base_index_s64(svptrue_b32(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svldnt1sw_gather_u64base_index_s64(svptrue_b32(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sh_gather_u64base_index_u64_with_svstnt1h_scatter_u64base_index_u64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svstnt1h_scatter_u64base_index_u64(svptrue_b16(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1sh_gather_u64base_index_u64(svptrue_b16(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1sw_gather_u64base_index_u64_with_svstnt1w_scatter_u64base_index_u64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svstnt1w_scatter_u64base_index_u64(svptrue_b32(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svldnt1sw_gather_u64base_index_u64(svptrue_b32(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1ub_gather_s64offset_s64_with_svstnt1b_scatter_s64offset_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_s64offset_s64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded =
        svldnt1ub_gather_s64offset_s64(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_s64offset_s64_with_svstnt1h_scatter_s64offset_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_s64offset_s64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1uh_gather_s64offset_s64(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_s64offset_s64_with_svstnt1w_scatter_s64offset_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 4u32.try_into().unwrap());
    svstnt1w_scatter_s64offset_s64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1uw_gather_s64offset_s64(svptrue_b32(), storage.as_ptr() as *const u32, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1ub_gather_s64offset_u64_with_svstnt1b_scatter_s64offset_u64() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_s64offset_u64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded =
        svldnt1ub_gather_s64offset_u64(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_s64offset_u64_with_svstnt1h_scatter_s64offset_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_s64offset_u64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svldnt1uh_gather_s64offset_u64(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_s64offset_u64_with_svstnt1w_scatter_s64offset_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_s64(0, 4u32.try_into().unwrap());
    svstnt1w_scatter_s64offset_u64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svldnt1uw_gather_s64offset_u64(svptrue_b32(), storage.as_ptr() as *const u32, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1ub_gather_u32offset_s32_with_svstnt1b_scatter_u32offset_s32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_u32offset_s32(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded =
        svldnt1ub_gather_u32offset_s32(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u32offset_s32_with_svstnt1h_scatter_u32offset_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u32offset_s32(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1uh_gather_u32offset_s32(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_i32(
        loaded,
        svindex_s32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1ub_gather_u32offset_u32_with_svstnt1b_scatter_u32offset_u32() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_u32offset_u32(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded =
        svldnt1ub_gather_u32offset_u32(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u32offset_u32_with_svstnt1h_scatter_u32offset_u32() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u32(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u32offset_u32(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svldnt1uh_gather_u32offset_u32(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_u32(
        loaded,
        svindex_u32((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1ub_gather_u64offset_s64_with_svstnt1b_scatter_u64offset_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_u64offset_s64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded =
        svldnt1ub_gather_u64offset_s64(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u64offset_s64_with_svstnt1h_scatter_u64offset_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u64offset_s64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1uh_gather_u64offset_s64(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_u64offset_s64_with_svstnt1w_scatter_u64offset_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    svstnt1w_scatter_u64offset_s64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1uw_gather_u64offset_s64(svptrue_b32(), storage.as_ptr() as *const u32, offsets);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1ub_gather_u64offset_u64_with_svstnt1b_scatter_u64offset_u64() {
    let mut storage = [0 as u8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_u64offset_u64(svptrue_b8(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u8 || val == i as u8);
    }
    svsetffr();
    let loaded =
        svldnt1ub_gather_u64offset_u64(svptrue_b8(), storage.as_ptr() as *const u8, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u64offset_u64_with_svstnt1h_scatter_u64offset_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u64offset_u64(svptrue_b16(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svldnt1uh_gather_u64offset_u64(svptrue_b16(), storage.as_ptr() as *const u16, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_u64offset_u64_with_svstnt1w_scatter_u64offset_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    svstnt1w_scatter_u64offset_u64(svptrue_b32(), storage.as_mut_ptr(), offsets, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svldnt1uw_gather_u64offset_u64(svptrue_b32(), storage.as_ptr() as *const u32, offsets);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1ub_gather_u32base_offset_s32_with_svstnt1b_scatter_u32base_offset_s32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_u32base_offset_s32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1ub_gather_u32base_offset_s32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u32base_offset_s32_with_svstnt1h_scatter_u32base_offset_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u32base_offset_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1uh_gather_u32base_offset_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1ub_gather_u32base_offset_u32_with_svstnt1b_scatter_u32base_offset_u32() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 1u32.try_into().unwrap());
    svstnt1b_scatter_u32base_offset_u32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1ub_gather_u32base_offset_u32(
        svptrue_b8(),
        bases,
        storage.as_ptr() as i64 + 1u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u32base_offset_u32_with_svstnt1h_scatter_u32base_offset_u32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u32base_offset_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1uh_gather_u32base_offset_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 + 2u32 as i64,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1ub_gather_u64base_offset_s64_with_svstnt1b_scatter_u64base_offset_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svstnt1b_scatter_u64base_offset_s64(svptrue_b8(), bases, 1u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1ub_gather_u64base_offset_s64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u64base_offset_s64_with_svstnt1h_scatter_u64base_offset_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svstnt1h_scatter_u64base_offset_s64(svptrue_b16(), bases, 2u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1uh_gather_u64base_offset_s64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_u64base_offset_s64_with_svstnt1w_scatter_u64base_offset_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svstnt1w_scatter_u64base_offset_s64(svptrue_b32(), bases, 4u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1uw_gather_u64base_offset_s64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1ub_gather_u64base_offset_u64_with_svstnt1b_scatter_u64base_offset_u64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svstnt1b_scatter_u64base_offset_u64(svptrue_b8(), bases, 1u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1ub_gather_u64base_offset_u64(svptrue_b8(), bases, 1u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u64base_offset_u64_with_svstnt1h_scatter_u64base_offset_u64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svstnt1h_scatter_u64base_offset_u64(svptrue_b16(), bases, 2u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1uh_gather_u64base_offset_u64(svptrue_b16(), bases, 2u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_u64base_offset_u64_with_svstnt1w_scatter_u64base_offset_u64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svstnt1w_scatter_u64base_offset_u64(svptrue_b32(), bases, 4u32.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1uw_gather_u64base_offset_u64(svptrue_b32(), bases, 4u32.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1ub_gather_u64base_s64_with_svstnt1b_scatter_u64base_s64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svstnt1b_scatter_u64base_s64(svptrue_b8(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1ub_gather_u64base_s64(svptrue_b8(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u64base_s64_with_svstnt1h_scatter_u64base_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svstnt1h_scatter_u64base_s64(svptrue_b16(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1uh_gather_u64base_s64(svptrue_b16(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_u64base_s64_with_svstnt1w_scatter_u64base_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svstnt1w_scatter_u64base_s64(svptrue_b32(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svldnt1uw_gather_u64base_s64(svptrue_b32(), bases);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1ub_gather_u64base_u64_with_svstnt1b_scatter_u64base_u64() {
    let mut storage = [0 as i8; 1280usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 1u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b8(), bases, offsets);
    svstnt1b_scatter_u64base_u64(svptrue_b8(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i8 || val == i as i8);
    }
    svsetffr();
    let loaded = svldnt1ub_gather_u64base_u64(svptrue_b8(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u64base_u64_with_svstnt1h_scatter_u64base_u64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svstnt1h_scatter_u64base_u64(svptrue_b16(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1uh_gather_u64base_u64(svptrue_b16(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_u64base_u64_with_svstnt1w_scatter_u64base_u64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svstnt1w_scatter_u64base_u64(svptrue_b32(), bases, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svldnt1uw_gather_u64base_u64(svptrue_b32(), bases);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_s64index_s64_with_svstnt1h_scatter_s64index_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svstnt1h_scatter_s64index_s64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1uh_gather_s64index_s64(svptrue_b16(), storage.as_ptr() as *const u16, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_s64index_s64_with_svstnt1w_scatter_s64index_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svstnt1w_scatter_s64index_s64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1uw_gather_s64index_s64(svptrue_b32(), storage.as_ptr() as *const u32, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_s64index_u64_with_svstnt1h_scatter_s64index_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svstnt1h_scatter_s64index_u64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svldnt1uh_gather_s64index_u64(svptrue_b16(), storage.as_ptr() as *const u16, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_s64index_u64_with_svstnt1w_scatter_s64index_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_s64(0, 1);
    svstnt1w_scatter_s64index_u64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svldnt1uw_gather_s64index_u64(svptrue_b32(), storage.as_ptr() as *const u32, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u64index_s64_with_svstnt1h_scatter_u64index_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svstnt1h_scatter_u64index_s64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded =
        svldnt1uh_gather_u64index_s64(svptrue_b16(), storage.as_ptr() as *const u16, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_u64index_s64_with_svstnt1w_scatter_u64index_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svstnt1w_scatter_u64index_s64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded =
        svldnt1uw_gather_u64index_s64(svptrue_b32(), storage.as_ptr() as *const u32, indices);
    assert_vector_matches_i64(
        loaded,
        svindex_s64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u64index_u64_with_svstnt1h_scatter_u64index_u64() {
    let mut storage = [0 as u16; 640usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svstnt1h_scatter_u64index_u64(svptrue_b16(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u16 || val == i as u16);
    }
    svsetffr();
    let loaded =
        svldnt1uh_gather_u64index_u64(svptrue_b16(), storage.as_ptr() as *const u16, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_u64index_u64_with_svstnt1w_scatter_u64index_u64() {
    let mut storage = [0 as u32; 320usize];
    let data = svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let indices = svindex_u64(0, 1);
    svstnt1w_scatter_u64index_u64(svptrue_b32(), storage.as_mut_ptr(), indices, data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as u32 || val == i as u32);
    }
    svsetffr();
    let loaded =
        svldnt1uw_gather_u64index_u64(svptrue_b32(), storage.as_ptr() as *const u32, indices);
    assert_vector_matches_u64(
        loaded,
        svindex_u64((0usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u32base_index_s32_with_svstnt1h_scatter_u32base_index_s32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u32base_index_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1uh_gather_u32base_index_s32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    assert_vector_matches_i32(
        loaded,
        svindex_s32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u32base_index_u32_with_svstnt1h_scatter_u32base_index_u32() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svindex_u32(0, 2u32.try_into().unwrap());
    svstnt1h_scatter_u32base_index_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
        data,
    );
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1uh_gather_u32base_index_u32(
        svptrue_b16(),
        bases,
        storage.as_ptr() as i64 / (2u32 as i64) + 1,
    );
    assert_vector_matches_u32(
        loaded,
        svindex_u32((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u64base_index_s64_with_svstnt1h_scatter_u64base_index_s64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svstnt1h_scatter_u64base_index_s64(svptrue_b16(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1uh_gather_u64base_index_s64(svptrue_b16(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_u64base_index_s64_with_svstnt1w_scatter_u64base_index_s64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svstnt1w_scatter_u64base_index_s64(svptrue_b32(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svldnt1uw_gather_u64base_index_s64(svptrue_b32(), bases, 1.try_into().unwrap());
    assert_vector_matches_i64(
        loaded,
        svindex_s64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uh_gather_u64base_index_u64_with_svstnt1h_scatter_u64base_index_u64() {
    let mut storage = [0 as i16; 640usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 2u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b16(), bases, offsets);
    svstnt1h_scatter_u64base_index_u64(svptrue_b16(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i16 || val == i as i16);
    }
    svsetffr();
    let loaded = svldnt1uh_gather_u64base_index_u64(svptrue_b16(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
#[simd_test(enable = "sve,sve2")]
unsafe fn test_svldnt1uw_gather_u64base_index_u64_with_svstnt1w_scatter_u64base_index_u64() {
    let mut storage = [0 as i32; 320usize];
    let data = svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap());
    let bases = svdup_n_u64(storage.as_ptr() as u64);
    let offsets = svindex_u64(0, 4u32.try_into().unwrap());
    let bases = svadd_u64_x(svptrue_b32(), bases, offsets);
    svstnt1w_scatter_u64base_index_u64(svptrue_b32(), bases, 1.try_into().unwrap(), data);
    for (i, &val) in storage.iter().enumerate() {
        assert!(val == 0 as i32 || val == i as i32);
    }
    svsetffr();
    let loaded = svldnt1uw_gather_u64base_index_u64(svptrue_b32(), bases, 1.try_into().unwrap());
    assert_vector_matches_u64(
        loaded,
        svindex_u64((1usize).try_into().unwrap(), 1usize.try_into().unwrap()),
    );
}
