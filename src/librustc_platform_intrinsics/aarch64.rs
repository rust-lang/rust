// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {Intrinsic, Type};
use IntrinsicDef::Named;

pub fn find(name: &str) -> Option<Intrinsic> {
    // if !name.starts_with("aarch64_v") { return None }
    // Some(match &name["aarch64_v".len()..] {
    intrinsics! {
        name, "aarch64_v",
        "ld2_s8" => Intrinsic {
            inputs: &[&Type::Pointer(&::I8, Some(&::I8x8), true)],
            output: &Type::Aggregate(false, &[&::I8x8, &::I8x8]),
            definition: Named("llvm.aarch64.neon.ld2.v8i8.p0v8i8")
        },
        "ld2_u8" => Intrinsic {
            inputs: &[&Type::Pointer(&::U8, Some(&::U8x8), true)],
            output: &Type::Aggregate(false, &[&::U8x8, &::U8x8]),
            definition: Named("llvm.aarch64.neon.ld2.v8i8.p0v8i8")
        },
        "ld2_s16" => Intrinsic {
            inputs: &[&Type::Pointer(&::I16, Some(&::I16x4), true)],
            output: &Type::Aggregate(false, &[&::I16x4, &::I16x4]),
            definition: Named("llvm.aarch64.neon.ld2.v4i16.p0v4i16")
        },
        "ld2_u16" => Intrinsic {
            inputs: &[&Type::Pointer(&::U16, Some(&::U16x4), true)],
            output: &Type::Aggregate(false, &[&::U16x4, &::U16x4]),
            definition: Named("llvm.aarch64.neon.ld2.v4i16.p0v4i16")
        },
        "ld2_s32" => Intrinsic {
            inputs: &[&Type::Pointer(&::I32, Some(&::I32x2), true)],
            output: &Type::Aggregate(false, &[&::I32x2, &::I32x2]),
            definition: Named("llvm.aarch64.neon.ld2.v2i32.p0v2i32")
        },
        "ld2_u32" => Intrinsic {
            inputs: &[&Type::Pointer(&::U32, Some(&::U32x2), true)],
            output: &Type::Aggregate(false, &[&::U32x2, &::U32x2]),
            definition: Named("llvm.aarch64.neon.ld2.v2i32.p0v2i32")
        },
        "ld2_s64" => Intrinsic {
            inputs: &[&Type::Pointer(&::I64, Some(&::I64x1), true)],
            output: &Type::Aggregate(false, &[&::I64x1, &::I64x1]),
            definition: Named("llvm.aarch64.neon.ld2.v1i64.p0v1i64")
        },
        "ld2_u64" => Intrinsic {
            inputs: &[&Type::Pointer(&::U64, Some(&::U64x1), true)],
            output: &Type::Aggregate(false, &[&::U64x1, &::U64x1]),
            definition: Named("llvm.aarch64.neon.ld2.v1i64.p0v1i64")
        },
        "ld2_f32" => Intrinsic {
            inputs: &[&Type::Pointer(&::F32, Some(&::F32x2), true)],
            output: &Type::Aggregate(false, &[&::F32x2, &::F32x2]),
            definition: Named("llvm.aarch64.neon.ld2.v2f32.p0v2f32")
        },
        "ld2_f64" => Intrinsic {
            inputs: &[&Type::Pointer(&::F64, Some(&::F64x1), true)],
            output: &Type::Aggregate(false, &[&::F64x1, &::F64x1]),
            definition: Named("llvm.aarch64.neon.ld2.v1f64.p0v1f64")
        },
        "ld2q_s8" => Intrinsic {
            inputs: &[&Type::Pointer(&::I8, Some(&::I8x16), true)],
            output: &Type::Aggregate(false, &[&::I8x16, &::I8x16]),
            definition: Named("llvm.aarch64.neon.ld2.v16i8.p0v16i8")
        },
        "ld2q_u8" => Intrinsic {
            inputs: &[&Type::Pointer(&::U8, Some(&::U8x16), true)],
            output: &Type::Aggregate(false, &[&::U8x16, &::U8x16]),
            definition: Named("llvm.aarch64.neon.ld2.v16i8.p0v16i8")
        },
        "ld2q_s16" => Intrinsic {
            inputs: &[&Type::Pointer(&::I16, Some(&::I16x8), true)],
            output: &Type::Aggregate(false, &[&::I16x8, &::I16x8]),
            definition: Named("llvm.aarch64.neon.ld2.v8i16.p0v8i16")
        },
        "ld2q_u16" => Intrinsic {
            inputs: &[&Type::Pointer(&::U16, Some(&::U16x8), true)],
            output: &Type::Aggregate(false, &[&::U16x8, &::U16x8]),
            definition: Named("llvm.aarch64.neon.ld2.v8i16.p0v8i16")
        },
        "ld2q_s32" => Intrinsic {
            inputs: &[&Type::Pointer(&::I32, Some(&::I32x4), true)],
            output: &Type::Aggregate(false, &[&::I32x4, &::I32x4]),
            definition: Named("llvm.aarch64.neon.ld2.v4i32.p0v4i32")
        },
        "ld2q_u32" => Intrinsic {
            inputs: &[&Type::Pointer(&::U32, Some(&::U32x4), true)],
            output: &Type::Aggregate(false, &[&::U32x4, &::U32x4]),
            definition: Named("llvm.aarch64.neon.ld2.v4i32.p0v4i32")
        },
        "ld2q_s64" => Intrinsic {
            inputs: &[&Type::Pointer(&::I64, Some(&::I64x2), true)],
            output: &Type::Aggregate(false, &[&::I64x2, &::I64x2]),
            definition: Named("llvm.aarch64.neon.ld2.v2i64.p0v2i64")
        },
        "ld2q_u64" => Intrinsic {
            inputs: &[&Type::Pointer(&::U64, Some(&::U64x2), true)],
            output: &Type::Aggregate(false, &[&::U64x2, &::U64x2]),
            definition: Named("llvm.aarch64.neon.ld2.v2i64.p0v2i64")
        },
        "ld2q_f32" => Intrinsic {
            inputs: &[&Type::Pointer(&::F32, Some(&::F32x4), true)],
            output: &Type::Aggregate(false, &[&::F32x4, &::F32x4]),
            definition: Named("llvm.aarch64.neon.ld2.v4f32.p0v4f32")
        },
        "ld2q_f64" => Intrinsic {
            inputs: &[&Type::Pointer(&::F64, Some(&::F64x2), true)],
            output: &Type::Aggregate(false, &[&::F64x2, &::F64x2]),
            definition: Named("llvm.aarch64.neon.ld2.v2f64.p0v2f64")
        },
        "ld3_s8" => Intrinsic {
            inputs: &[&Type::Pointer(&::I8, Some(&::I8x8), true)],
            output: &Type::Aggregate(false, &[&::I8x8, &::I8x8, &::I8x8]),
            definition: Named("llvm.aarch64.neon.ld3.v8i8.p0v8i8")
        },
        "ld3_u8" => Intrinsic {
            inputs: &[&Type::Pointer(&::U8, Some(&::U8x8), true)],
            output: &Type::Aggregate(false, &[&::U8x8, &::U8x8, &::U8x8]),
            definition: Named("llvm.aarch64.neon.ld3.v8i8.p0v8i8")
        },
        "ld3_s16" => Intrinsic {
            inputs: &[&Type::Pointer(&::I16, Some(&::I16x4), true)],
            output: &Type::Aggregate(false, &[&::I16x4, &::I16x4, &::I16x4]),
            definition: Named("llvm.aarch64.neon.ld3.v4i16.p0v4i16")
        },
        "ld3_u16" => Intrinsic {
            inputs: &[&Type::Pointer(&::U16, Some(&::U16x4), true)],
            output: &Type::Aggregate(false, &[&::U16x4, &::U16x4, &::U16x4]),
            definition: Named("llvm.aarch64.neon.ld3.v4i16.p0v4i16")
        },
        "ld3_s32" => Intrinsic {
            inputs: &[&Type::Pointer(&::I32, Some(&::I32x2), true)],
            output: &Type::Aggregate(false, &[&::I32x2, &::I32x2, &::I32x2]),
            definition: Named("llvm.aarch64.neon.ld3.v2i32.p0v2i32")
        },
        "ld3_u32" => Intrinsic {
            inputs: &[&Type::Pointer(&::U32, Some(&::U32x2), true)],
            output: &Type::Aggregate(false, &[&::U32x2, &::U32x2, &::U32x2]),
            definition: Named("llvm.aarch64.neon.ld3.v2i32.p0v2i32")
        },
        "ld3_s64" => Intrinsic {
            inputs: &[&Type::Pointer(&::I64, Some(&::I64x1), true)],
            output: &Type::Aggregate(false, &[&::I64x1, &::I64x1, &::I64x1]),
            definition: Named("llvm.aarch64.neon.ld3.v1i64.p0v1i64")
        },
        "ld3_u64" => Intrinsic {
            inputs: &[&Type::Pointer(&::U64, Some(&::U64x1), true)],
            output: &Type::Aggregate(false, &[&::U64x1, &::U64x1, &::U64x1]),
            definition: Named("llvm.aarch64.neon.ld3.v1i64.p0v1i64")
        },
        "ld3_f32" => Intrinsic {
            inputs: &[&Type::Pointer(&::F32, Some(&::F32x2), true)],
            output: &Type::Aggregate(false, &[&::F32x2, &::F32x2, &::F32x2]),
            definition: Named("llvm.aarch64.neon.ld3.v2f32.p0v2f32")
        },
        "ld3_f64" => Intrinsic {
            inputs: &[&Type::Pointer(&::F64, Some(&::F64x1), true)],
            output: &Type::Aggregate(false, &[&::F64x1, &::F64x1, &::F64x1]),
            definition: Named("llvm.aarch64.neon.ld3.v1f64.p0v1f64")
        },
        "ld3q_s8" => Intrinsic {
            inputs: &[&Type::Pointer(&::I8, Some(&::I8x16), true)],
            output: &Type::Aggregate(false, &[&::I8x16, &::I8x16, &::I8x16]),
            definition: Named("llvm.aarch64.neon.ld3.v16i8.p0v16i8")
        },
        "ld3q_u8" => Intrinsic {
            inputs: &[&Type::Pointer(&::U8, Some(&::U8x16), true)],
            output: &Type::Aggregate(false, &[&::U8x16, &::U8x16, &::U8x16]),
            definition: Named("llvm.aarch64.neon.ld3.v16i8.p0v16i8")
        },
        "ld3q_s16" => Intrinsic {
            inputs: &[&Type::Pointer(&::I16, Some(&::I16x8), true)],
            output: &Type::Aggregate(false, &[&::I16x8, &::I16x8, &::I16x8]),
            definition: Named("llvm.aarch64.neon.ld3.v8i16.p0v8i16")
        },
        "ld3q_u16" => Intrinsic {
            inputs: &[&Type::Pointer(&::U16, Some(&::U16x8), true)],
            output: &Type::Aggregate(false, &[&::U16x8, &::U16x8, &::U16x8]),
            definition: Named("llvm.aarch64.neon.ld3.v8i16.p0v8i16")
        },
        "ld3q_s32" => Intrinsic {
            inputs: &[&Type::Pointer(&::I32, Some(&::I32x4), true)],
            output: &Type::Aggregate(false, &[&::I32x4, &::I32x4, &::I32x4]),
            definition: Named("llvm.aarch64.neon.ld3.v4i32.p0v4i32")
        },
        "ld3q_u32" => Intrinsic {
            inputs: &[&Type::Pointer(&::U32, Some(&::U32x4), true)],
            output: &Type::Aggregate(false, &[&::U32x4, &::U32x4, &::U32x4]),
            definition: Named("llvm.aarch64.neon.ld3.v4i32.p0v4i32")
        },
        "ld3q_s64" => Intrinsic {
            inputs: &[&Type::Pointer(&::I64, Some(&::I64x2), true)],
            output: &Type::Aggregate(false, &[&::I64x2, &::I64x2, &::I64x2]),
            definition: Named("llvm.aarch64.neon.ld3.v2i64.p0v2i64")
        },
        "ld3q_u64" => Intrinsic {
            inputs: &[&Type::Pointer(&::U64, Some(&::U64x2), true)],
            output: &Type::Aggregate(false, &[&::U64x2, &::U64x2, &::U64x2]),
            definition: Named("llvm.aarch64.neon.ld3.v2i64.p0v2i64")
        },
        "ld3q_f32" => Intrinsic {
            inputs: &[&Type::Pointer(&::F32, Some(&::F32x4), true)],
            output: &Type::Aggregate(false, &[&::F32x4, &::F32x4, &::F32x4]),
            definition: Named("llvm.aarch64.neon.ld3.v4f32.p0v4f32")
        },
        "ld3q_f64" => Intrinsic {
            inputs: &[&Type::Pointer(&::F64, Some(&::F64x2), true)],
            output: &Type::Aggregate(false, &[&::F64x2, &::F64x2, &::F64x2]),
            definition: Named("llvm.aarch64.neon.ld3.v2f64.p0v2f64")
        },
        "ld4_s8" => Intrinsic {
            inputs: &[&Type::Pointer(&::I8, Some(&::I8x8), true)],
            output: &Type::Aggregate(false, &[&::I8x8, &::I8x8, &::I8x8, &::I8x8]),
            definition: Named("llvm.aarch64.neon.ld4.v8i8.p0v8i8")
        },
        "ld4_u8" => Intrinsic {
            inputs: &[&Type::Pointer(&::U8, Some(&::U8x8), true)],
            output: &Type::Aggregate(false, &[&::U8x8, &::U8x8, &::U8x8, &::U8x8]),
            definition: Named("llvm.aarch64.neon.ld4.v8i8.p0v8i8")
        },
        "ld4_s16" => Intrinsic {
            inputs: &[&Type::Pointer(&::I16, Some(&::I16x4), true)],
            output: &Type::Aggregate(false, &[&::I16x4, &::I16x4, &::I16x4, &::I16x4]),
            definition: Named("llvm.aarch64.neon.ld4.v4i16.p0v4i16")
        },
        "ld4_u16" => Intrinsic {
            inputs: &[&Type::Pointer(&::U16, Some(&::U16x4), true)],
            output: &Type::Aggregate(false, &[&::U16x4, &::U16x4, &::U16x4, &::U16x4]),
            definition: Named("llvm.aarch64.neon.ld4.v4i16.p0v4i16")
        },
        "ld4_s32" => Intrinsic {
            inputs: &[&Type::Pointer(&::I32, Some(&::I32x2), true)],
            output: &Type::Aggregate(false, &[&::I32x2, &::I32x2, &::I32x2, &::I32x2]),
            definition: Named("llvm.aarch64.neon.ld4.v2i32.p0v2i32")
        },
        "ld4_u32" => Intrinsic {
            inputs: &[&Type::Pointer(&::U32, Some(&::U32x2), true)],
            output: &Type::Aggregate(false, &[&::U32x2, &::U32x2, &::U32x2, &::U32x2]),
            definition: Named("llvm.aarch64.neon.ld4.v2i32.p0v2i32")
        },
        "ld4_s64" => Intrinsic {
            inputs: &[&Type::Pointer(&::I64, Some(&::I64x1), true)],
            output: &Type::Aggregate(false, &[&::I64x1, &::I64x1, &::I64x1, &::I64x1]),
            definition: Named("llvm.aarch64.neon.ld4.v1i64.p0v1i64")
        },
        "ld4_u64" => Intrinsic {
            inputs: &[&Type::Pointer(&::U64, Some(&::U64x1), true)],
            output: &Type::Aggregate(false, &[&::U64x1, &::U64x1, &::U64x1, &::U64x1]),
            definition: Named("llvm.aarch64.neon.ld4.v1i64.p0v1i64")
        },
        "ld4_f32" => Intrinsic {
            inputs: &[&Type::Pointer(&::F32, Some(&::F32x2), true)],
            output: &Type::Aggregate(false, &[&::F32x2, &::F32x2, &::F32x2, &::F32x2]),
            definition: Named("llvm.aarch64.neon.ld4.v2f32.p0v2f32")
        },
        "ld4_f64" => Intrinsic {
            inputs: &[&Type::Pointer(&::F64, Some(&::F64x1), true)],
            output: &Type::Aggregate(false, &[&::F64x1, &::F64x1, &::F64x1, &::F64x1]),
            definition: Named("llvm.aarch64.neon.ld4.v1f64.p0v1f64")
        },
        "ld4q_s8" => Intrinsic {
            inputs: &[&Type::Pointer(&::I8, Some(&::I8x16), true)],
            output: &Type::Aggregate(false, &[&::I8x16, &::I8x16, &::I8x16, &::I8x16]),
            definition: Named("llvm.aarch64.neon.ld4.v16i8.p0v16i8")
        },
        "ld4q_u8" => Intrinsic {
            inputs: &[&Type::Pointer(&::U8, Some(&::U8x16), true)],
            output: &Type::Aggregate(false, &[&::U8x16, &::U8x16, &::U8x16, &::U8x16]),
            definition: Named("llvm.aarch64.neon.ld4.v16i8.p0v16i8")
        },
        "ld4q_s16" => Intrinsic {
            inputs: &[&Type::Pointer(&::I16, Some(&::I16x8), true)],
            output: &Type::Aggregate(false, &[&::I16x8, &::I16x8, &::I16x8, &::I16x8]),
            definition: Named("llvm.aarch64.neon.ld4.v8i16.p0v8i16")
        },
        "ld4q_u16" => Intrinsic {
            inputs: &[&Type::Pointer(&::U16, Some(&::U16x8), true)],
            output: &Type::Aggregate(false, &[&::U16x8, &::U16x8, &::U16x8, &::U16x8]),
            definition: Named("llvm.aarch64.neon.ld4.v8i16.p0v8i16")
        },
        "ld4q_s32" => Intrinsic {
            inputs: &[&Type::Pointer(&::I32, Some(&::I32x4), true)],
            output: &Type::Aggregate(false, &[&::I32x4, &::I32x4, &::I32x4, &::I32x4]),
            definition: Named("llvm.aarch64.neon.ld4.v4i32.p0v4i32")
        },
        "ld4q_u32" => Intrinsic {
            inputs: &[&Type::Pointer(&::U32, Some(&::U32x4), true)],
            output: &Type::Aggregate(false, &[&::U32x4, &::U32x4, &::U32x4, &::U32x4]),
            definition: Named("llvm.aarch64.neon.ld4.v4i32.p0v4i32")
        },
        "ld4q_s64" => Intrinsic {
            inputs: &[&Type::Pointer(&::I64, Some(&::I64x2), true)],
            output: &Type::Aggregate(false, &[&::I64x2, &::I64x2, &::I64x2, &::I64x2]),
            definition: Named("llvm.aarch64.neon.ld4.v2i64.p0v2i64")
        },
        "ld4q_u64" => Intrinsic {
            inputs: &[&Type::Pointer(&::U64, Some(&::U64x2), true)],
            output: &Type::Aggregate(false, &[&::U64x2, &::U64x2, &::U64x2, &::U64x2]),
            definition: Named("llvm.aarch64.neon.ld4.v2i64.p0v2i64")
        },
        "ld4q_f32" => Intrinsic {
            inputs: &[&Type::Pointer(&::F32, Some(&::F32x4), true)],
            output: &Type::Aggregate(false, &[&::F32x4, &::F32x4, &::F32x4, &::F32x4]),
            definition: Named("llvm.aarch64.neon.ld4.v4f32.p0v4f32")
        },
        "ld4q_f64" => Intrinsic {
            inputs: &[&Type::Pointer(&::F64, Some(&::F64x2), true)],
            output: &Type::Aggregate(false, &[&::F64x2, &::F64x2, &::F64x2, &::F64x2]),
            definition: Named("llvm.aarch64.neon.ld4.v2f64.p0v2f64")
        },
        "ld2_dup_s8" => Intrinsic {
            inputs: &[&Type::Pointer(&::I8, None, true)],
            output: &Type::Aggregate(false, &[&::I8x8, &::I8x8]),
            definition: Named("llvm.aarch64.neon.ld2.v8i8.p0i8")
        },
        "ld2_dup_u8" => Intrinsic {
            inputs: &[&Type::Pointer(&::U8, None, true)],
            output: &Type::Aggregate(false, &[&::U8x8, &::U8x8]),
            definition: Named("llvm.aarch64.neon.ld2.v8i8.p0i8")
        },
        "ld2_dup_s16" => Intrinsic {
            inputs: &[&Type::Pointer(&::I16, None, true)],
            output: &Type::Aggregate(false, &[&::I16x4, &::I16x4]),
            definition: Named("llvm.aarch64.neon.ld2.v4i16.p0i16")
        },
        "ld2_dup_u16" => Intrinsic {
            inputs: &[&Type::Pointer(&::U16, None, true)],
            output: &Type::Aggregate(false, &[&::U16x4, &::U16x4]),
            definition: Named("llvm.aarch64.neon.ld2.v4i16.p0i16")
        },
        "ld2_dup_s32" => Intrinsic {
            inputs: &[&Type::Pointer(&::I32, None, true)],
            output: &Type::Aggregate(false, &[&::I32x2, &::I32x2]),
            definition: Named("llvm.aarch64.neon.ld2.v2i32.p0i32")
        },
        "ld2_dup_u32" => Intrinsic {
            inputs: &[&Type::Pointer(&::U32, None, true)],
            output: &Type::Aggregate(false, &[&::U32x2, &::U32x2]),
            definition: Named("llvm.aarch64.neon.ld2.v2i32.p0i32")
        },
        "ld2_dup_s64" => Intrinsic {
            inputs: &[&Type::Pointer(&::I64, None, true)],
            output: &Type::Aggregate(false, &[&::I64x1, &::I64x1]),
            definition: Named("llvm.aarch64.neon.ld2.v1i64.p0i64")
        },
        "ld2_dup_u64" => Intrinsic {
            inputs: &[&Type::Pointer(&::U64, None, true)],
            output: &Type::Aggregate(false, &[&::U64x1, &::U64x1]),
            definition: Named("llvm.aarch64.neon.ld2.v1i64.p0i64")
        },
        "ld2_dup_f32" => Intrinsic {
            inputs: &[&Type::Pointer(&::F32, None, true)],
            output: &Type::Aggregate(false, &[&::F32x2, &::F32x2]),
            definition: Named("llvm.aarch64.neon.ld2.v2f32.p0f32")
        },
        "ld2_dup_f64" => Intrinsic {
            inputs: &[&Type::Pointer(&::F64, None, true)],
            output: &Type::Aggregate(false, &[&::F64x1, &::F64x1]),
            definition: Named("llvm.aarch64.neon.ld2.v1f64.p0f64")
        },
        "ld2q_dup_s8" => Intrinsic {
            inputs: &[&Type::Pointer(&::I8, None, true)],
            output: &Type::Aggregate(false, &[&::I8x16, &::I8x16]),
            definition: Named("llvm.aarch64.neon.ld2.v16i8.p0i8")
        },
        "ld2q_dup_u8" => Intrinsic {
            inputs: &[&Type::Pointer(&::U8, None, true)],
            output: &Type::Aggregate(false, &[&::U8x16, &::U8x16]),
            definition: Named("llvm.aarch64.neon.ld2.v16i8.p0i8")
        },
        "ld2q_dup_s16" => Intrinsic {
            inputs: &[&Type::Pointer(&::I16, None, true)],
            output: &Type::Aggregate(false, &[&::I16x8, &::I16x8]),
            definition: Named("llvm.aarch64.neon.ld2.v8i16.p0i16")
        },
        "ld2q_dup_u16" => Intrinsic {
            inputs: &[&Type::Pointer(&::U16, None, true)],
            output: &Type::Aggregate(false, &[&::U16x8, &::U16x8]),
            definition: Named("llvm.aarch64.neon.ld2.v8i16.p0i16")
        },
        "ld2q_dup_s32" => Intrinsic {
            inputs: &[&Type::Pointer(&::I32, None, true)],
            output: &Type::Aggregate(false, &[&::I32x4, &::I32x4]),
            definition: Named("llvm.aarch64.neon.ld2.v4i32.p0i32")
        },
        "ld2q_dup_u32" => Intrinsic {
            inputs: &[&Type::Pointer(&::U32, None, true)],
            output: &Type::Aggregate(false, &[&::U32x4, &::U32x4]),
            definition: Named("llvm.aarch64.neon.ld2.v4i32.p0i32")
        },
        "ld2q_dup_s64" => Intrinsic {
            inputs: &[&Type::Pointer(&::I64, None, true)],
            output: &Type::Aggregate(false, &[&::I64x2, &::I64x2]),
            definition: Named("llvm.aarch64.neon.ld2.v2i64.p0i64")
        },
        "ld2q_dup_u64" => Intrinsic {
            inputs: &[&Type::Pointer(&::U64, None, true)],
            output: &Type::Aggregate(false, &[&::U64x2, &::U64x2]),
            definition: Named("llvm.aarch64.neon.ld2.v2i64.p0i64")
        },
        "ld2q_dup_f32" => Intrinsic {
            inputs: &[&Type::Pointer(&::F32, None, true)],
            output: &Type::Aggregate(false, &[&::F32x4, &::F32x4]),
            definition: Named("llvm.aarch64.neon.ld2.v4f32.p0f32")
        },
        "ld2q_dup_f64" => Intrinsic {
            inputs: &[&Type::Pointer(&::F64, None, true)],
            output: &Type::Aggregate(false, &[&::F64x2, &::F64x2]),
            definition: Named("llvm.aarch64.neon.ld2.v2f64.p0f64")
        },
        "ld3_dup_s8" => Intrinsic {
            inputs: &[&Type::Pointer(&::I8, None, true)],
            output: &Type::Aggregate(false, &[&::I8x8, &::I8x8, &::I8x8]),
            definition: Named("llvm.aarch64.neon.ld3.v8i8.p0i8")
        },
        "ld3_dup_u8" => Intrinsic {
            inputs: &[&Type::Pointer(&::U8, None, true)],
            output: &Type::Aggregate(false, &[&::U8x8, &::U8x8, &::U8x8]),
            definition: Named("llvm.aarch64.neon.ld3.v8i8.p0i8")
        },
        "ld3_dup_s16" => Intrinsic {
            inputs: &[&Type::Pointer(&::I16, None, true)],
            output: &Type::Aggregate(false, &[&::I16x4, &::I16x4, &::I16x4]),
            definition: Named("llvm.aarch64.neon.ld3.v4i16.p0i16")
        },
        "ld3_dup_u16" => Intrinsic {
            inputs: &[&Type::Pointer(&::U16, None, true)],
            output: &Type::Aggregate(false, &[&::U16x4, &::U16x4, &::U16x4]),
            definition: Named("llvm.aarch64.neon.ld3.v4i16.p0i16")
        },
        "ld3_dup_s32" => Intrinsic {
            inputs: &[&Type::Pointer(&::I32, None, true)],
            output: &Type::Aggregate(false, &[&::I32x2, &::I32x2, &::I32x2]),
            definition: Named("llvm.aarch64.neon.ld3.v2i32.p0i32")
        },
        "ld3_dup_u32" => Intrinsic {
            inputs: &[&Type::Pointer(&::U32, None, true)],
            output: &Type::Aggregate(false, &[&::U32x2, &::U32x2, &::U32x2]),
            definition: Named("llvm.aarch64.neon.ld3.v2i32.p0i32")
        },
        "ld3_dup_s64" => Intrinsic {
            inputs: &[&Type::Pointer(&::I64, None, true)],
            output: &Type::Aggregate(false, &[&::I64x1, &::I64x1, &::I64x1]),
            definition: Named("llvm.aarch64.neon.ld3.v1i64.p0i64")
        },
        "ld3_dup_u64" => Intrinsic {
            inputs: &[&Type::Pointer(&::U64, None, true)],
            output: &Type::Aggregate(false, &[&::U64x1, &::U64x1, &::U64x1]),
            definition: Named("llvm.aarch64.neon.ld3.v1i64.p0i64")
        },
        "ld3_dup_f32" => Intrinsic {
            inputs: &[&Type::Pointer(&::F32, None, true)],
            output: &Type::Aggregate(false, &[&::F32x2, &::F32x2, &::F32x2]),
            definition: Named("llvm.aarch64.neon.ld3.v2f32.p0f32")
        },
        "ld3_dup_f64" => Intrinsic {
            inputs: &[&Type::Pointer(&::F64, None, true)],
            output: &Type::Aggregate(false, &[&::F64x1, &::F64x1, &::F64x1]),
            definition: Named("llvm.aarch64.neon.ld3.v1f64.p0f64")
        },
        "ld3q_dup_s8" => Intrinsic {
            inputs: &[&Type::Pointer(&::I8, None, true)],
            output: &Type::Aggregate(false, &[&::I8x16, &::I8x16, &::I8x16]),
            definition: Named("llvm.aarch64.neon.ld3.v16i8.p0i8")
        },
        "ld3q_dup_u8" => Intrinsic {
            inputs: &[&Type::Pointer(&::U8, None, true)],
            output: &Type::Aggregate(false, &[&::U8x16, &::U8x16, &::U8x16]),
            definition: Named("llvm.aarch64.neon.ld3.v16i8.p0i8")
        },
        "ld3q_dup_s16" => Intrinsic {
            inputs: &[&Type::Pointer(&::I16, None, true)],
            output: &Type::Aggregate(false, &[&::I16x8, &::I16x8, &::I16x8]),
            definition: Named("llvm.aarch64.neon.ld3.v8i16.p0i16")
        },
        "ld3q_dup_u16" => Intrinsic {
            inputs: &[&Type::Pointer(&::U16, None, true)],
            output: &Type::Aggregate(false, &[&::U16x8, &::U16x8, &::U16x8]),
            definition: Named("llvm.aarch64.neon.ld3.v8i16.p0i16")
        },
        "ld3q_dup_s32" => Intrinsic {
            inputs: &[&Type::Pointer(&::I32, None, true)],
            output: &Type::Aggregate(false, &[&::I32x4, &::I32x4, &::I32x4]),
            definition: Named("llvm.aarch64.neon.ld3.v4i32.p0i32")
        },
        "ld3q_dup_u32" => Intrinsic {
            inputs: &[&Type::Pointer(&::U32, None, true)],
            output: &Type::Aggregate(false, &[&::U32x4, &::U32x4, &::U32x4]),
            definition: Named("llvm.aarch64.neon.ld3.v4i32.p0i32")
        },
        "ld3q_dup_s64" => Intrinsic {
            inputs: &[&Type::Pointer(&::I64, None, true)],
            output: &Type::Aggregate(false, &[&::I64x2, &::I64x2, &::I64x2]),
            definition: Named("llvm.aarch64.neon.ld3.v2i64.p0i64")
        },
        "ld3q_dup_u64" => Intrinsic {
            inputs: &[&Type::Pointer(&::U64, None, true)],
            output: &Type::Aggregate(false, &[&::U64x2, &::U64x2, &::U64x2]),
            definition: Named("llvm.aarch64.neon.ld3.v2i64.p0i64")
        },
        "ld3q_dup_f32" => Intrinsic {
            inputs: &[&Type::Pointer(&::F32, None, true)],
            output: &Type::Aggregate(false, &[&::F32x4, &::F32x4, &::F32x4]),
            definition: Named("llvm.aarch64.neon.ld3.v4f32.p0f32")
        },
        "ld3q_dup_f64" => Intrinsic {
            inputs: &[&Type::Pointer(&::F64, None, true)],
            output: &Type::Aggregate(false, &[&::F64x2, &::F64x2, &::F64x2]),
            definition: Named("llvm.aarch64.neon.ld3.v2f64.p0f64")
        },
        "ld4_dup_s8" => Intrinsic {
            inputs: &[&Type::Pointer(&::I8, None, true)],
            output: &Type::Aggregate(false, &[&::I8x8, &::I8x8, &::I8x8, &::I8x8]),
            definition: Named("llvm.aarch64.neon.ld4.v8i8.p0i8")
        },
        "ld4_dup_u8" => Intrinsic {
            inputs: &[&Type::Pointer(&::U8, None, true)],
            output: &Type::Aggregate(false, &[&::U8x8, &::U8x8, &::U8x8, &::U8x8]),
            definition: Named("llvm.aarch64.neon.ld4.v8i8.p0i8")
        },
        "ld4_dup_s16" => Intrinsic {
            inputs: &[&Type::Pointer(&::I16, None, true)],
            output: &Type::Aggregate(false, &[&::I16x4, &::I16x4, &::I16x4, &::I16x4]),
            definition: Named("llvm.aarch64.neon.ld4.v4i16.p0i16")
        },
        "ld4_dup_u16" => Intrinsic {
            inputs: &[&Type::Pointer(&::U16, None, true)],
            output: &Type::Aggregate(false, &[&::U16x4, &::U16x4, &::U16x4, &::U16x4]),
            definition: Named("llvm.aarch64.neon.ld4.v4i16.p0i16")
        },
        "ld4_dup_s32" => Intrinsic {
            inputs: &[&Type::Pointer(&::I32, None, true)],
            output: &Type::Aggregate(false, &[&::I32x2, &::I32x2, &::I32x2, &::I32x2]),
            definition: Named("llvm.aarch64.neon.ld4.v2i32.p0i32")
        },
        "ld4_dup_u32" => Intrinsic {
            inputs: &[&Type::Pointer(&::U32, None, true)],
            output: &Type::Aggregate(false, &[&::U32x2, &::U32x2, &::U32x2, &::U32x2]),
            definition: Named("llvm.aarch64.neon.ld4.v2i32.p0i32")
        },
        "ld4_dup_s64" => Intrinsic {
            inputs: &[&Type::Pointer(&::I64, None, true)],
            output: &Type::Aggregate(false, &[&::I64x1, &::I64x1, &::I64x1, &::I64x1]),
            definition: Named("llvm.aarch64.neon.ld4.v1i64.p0i64")
        },
        "ld4_dup_u64" => Intrinsic {
            inputs: &[&Type::Pointer(&::U64, None, true)],
            output: &Type::Aggregate(false, &[&::U64x1, &::U64x1, &::U64x1, &::U64x1]),
            definition: Named("llvm.aarch64.neon.ld4.v1i64.p0i64")
        },
        "ld4_dup_f32" => Intrinsic {
            inputs: &[&Type::Pointer(&::F32, None, true)],
            output: &Type::Aggregate(false, &[&::F32x2, &::F32x2, &::F32x2, &::F32x2]),
            definition: Named("llvm.aarch64.neon.ld4.v2f32.p0f32")
        },
        "ld4_dup_f64" => Intrinsic {
            inputs: &[&Type::Pointer(&::F64, None, true)],
            output: &Type::Aggregate(false, &[&::F64x1, &::F64x1, &::F64x1, &::F64x1]),
            definition: Named("llvm.aarch64.neon.ld4.v1f64.p0f64")
        },
        "ld4q_dup_s8" => Intrinsic {
            inputs: &[&Type::Pointer(&::I8, None, true)],
            output: &Type::Aggregate(false, &[&::I8x16, &::I8x16, &::I8x16, &::I8x16]),
            definition: Named("llvm.aarch64.neon.ld4.v16i8.p0i8")
        },
        "ld4q_dup_u8" => Intrinsic {
            inputs: &[&Type::Pointer(&::U8, None, true)],
            output: &Type::Aggregate(false, &[&::U8x16, &::U8x16, &::U8x16, &::U8x16]),
            definition: Named("llvm.aarch64.neon.ld4.v16i8.p0i8")
        },
        "ld4q_dup_s16" => Intrinsic {
            inputs: &[&Type::Pointer(&::I16, None, true)],
            output: &Type::Aggregate(false, &[&::I16x8, &::I16x8, &::I16x8, &::I16x8]),
            definition: Named("llvm.aarch64.neon.ld4.v8i16.p0i16")
        },
        "ld4q_dup_u16" => Intrinsic {
            inputs: &[&Type::Pointer(&::U16, None, true)],
            output: &Type::Aggregate(false, &[&::U16x8, &::U16x8, &::U16x8, &::U16x8]),
            definition: Named("llvm.aarch64.neon.ld4.v8i16.p0i16")
        },
        "ld4q_dup_s32" => Intrinsic {
            inputs: &[&Type::Pointer(&::I32, None, true)],
            output: &Type::Aggregate(false, &[&::I32x4, &::I32x4, &::I32x4, &::I32x4]),
            definition: Named("llvm.aarch64.neon.ld4.v4i32.p0i32")
        },
        "ld4q_dup_u32" => Intrinsic {
            inputs: &[&Type::Pointer(&::U32, None, true)],
            output: &Type::Aggregate(false, &[&::U32x4, &::U32x4, &::U32x4, &::U32x4]),
            definition: Named("llvm.aarch64.neon.ld4.v4i32.p0i32")
        },
        "ld4q_dup_s64" => Intrinsic {
            inputs: &[&Type::Pointer(&::I64, None, true)],
            output: &Type::Aggregate(false, &[&::I64x2, &::I64x2, &::I64x2, &::I64x2]),
            definition: Named("llvm.aarch64.neon.ld4.v2i64.p0i64")
        },
        "ld4q_dup_u64" => Intrinsic {
            inputs: &[&Type::Pointer(&::U64, None, true)],
            output: &Type::Aggregate(false, &[&::U64x2, &::U64x2, &::U64x2, &::U64x2]),
            definition: Named("llvm.aarch64.neon.ld4.v2i64.p0i64")
        },
        "ld4q_dup_f32" => Intrinsic {
            inputs: &[&Type::Pointer(&::F32, None, true)],
            output: &Type::Aggregate(false, &[&::F32x4, &::F32x4, &::F32x4, &::F32x4]),
            definition: Named("llvm.aarch64.neon.ld4.v4f32.p0f32")
        },
        "ld4q_dup_f64" => Intrinsic {
            inputs: &[&Type::Pointer(&::F64, None, true)],
            output: &Type::Aggregate(false, &[&::F64x2, &::F64x2, &::F64x2, &::F64x2]),
            definition: Named("llvm.aarch64.neon.ld4.v2f64.p0f64")
        },
    }
}
