// ignore-tidy-linelength
//@ only-aarch64
//@ compile-flags: -Cdebuginfo=2 -Copt-level=0
//@ revisions: POST-LLVM-22 PRE-LLVM-22
//@ [PRE-LLVM-22] max-llvm-major-version: 21
//@ [POST-LLVM-22] min-llvm-version: 22

#![crate_type = "lib"]
#![allow(incomplete_features, internal_features)]
#![feature(rustc_attrs)]

// Test that we generate the correct debuginfo for scalable vector types.

#[rustc_scalable_vector(16)]
#[allow(non_camel_case_types)]
struct svbool_t(bool);

#[rustc_scalable_vector(16)]
#[allow(non_camel_case_types)]
struct svint8_t(i8);

#[rustc_scalable_vector(16)]
#[allow(non_camel_case_types)]
struct svuint8_t(u8);

#[rustc_scalable_vector(8)]
#[allow(non_camel_case_types)]
struct svint16_t(i16);

#[rustc_scalable_vector(8)]
#[allow(non_camel_case_types)]
struct svuint16_t(u16);

#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
struct svint32_t(i32);

#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
struct svuint32_t(u32);

#[rustc_scalable_vector(2)]
#[allow(non_camel_case_types)]
struct svint64_t(i64);

#[rustc_scalable_vector(2)]
#[allow(non_camel_case_types)]
struct svuint64_t(u64);

#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
struct svfloat32_t(f32);

#[rustc_scalable_vector(2)]
#[allow(non_camel_case_types)]
struct svfloat64_t(f64);

#[target_feature(enable = "sve")]
pub fn locals() {
    // CHECK-DAG: name: "svbool_t",{{.*}}, baseType: ![[CT1:[0-9]+]]
    // PRE-LLVM-22-DAG: ![[CT1]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTYU8:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS8:[0-9]+]])
    // POST-LLVM-22-DAG: ![[CT1]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTYU8:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS8:[0-9]+]], bitStride: i64 1)
    // CHECK-DAG: ![[ELTTYU8]] = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)
    // CHECK-DAG: ![[ELTS8]] = !{![[REALELTS8:[0-9]+]]}
    // CHECK-DAG: ![[REALELTS8]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 8, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
    let b8: svbool_t;

    // CHECK-DAG: name: "svint8_t",{{.*}}, baseType: ![[CT8:[0-9]+]]
    // CHECK-DAG: ![[CT8]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTYS8:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS8:[0-9]+]])
    // CHECK-DAG: ![[ELTTYS8]] = !DIBasicType(name: "i8", size: 8, encoding: DW_ATE_signed)
    let s8: svint8_t;

    // PRE-LLVM-22-DAG: name: "svuint8_t",{{.*}}, baseType: ![[CT1:[0-9]+]]
    // POST-LLVM-22-DAG: name: "svuint8_t",{{.*}}, baseType: ![[CT8:[0-9]+]]
    // POST-LLVM-22-DAG: ![[CT8]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTYU8]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS8]])
    let u8: svuint8_t;

    // CHECK-DAG: name: "svint16_t",{{.*}}, baseType: ![[CT16:[0-9]+]]
    // CHECK-DAG: ![[CT16]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY16:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS16:[0-9]+]])
    // CHECK-DAG: ![[ELTTY16]] = !DIBasicType(name: "i16", size: 16, encoding: DW_ATE_signed)
    // CHECK-DAG: ![[ELTS16]] = !{![[REALELTS16:[0-9]+]]}
    // CHECK-DAG: ![[REALELTS16]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 4, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
    let s16: svint16_t;

    // CHECK-DAG: name: "svuint16_t",{{.*}}, baseType: ![[CT16:[0-9]+]]
    // CHECK-DAG: ![[CT16]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY16:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS16]])
    // CHECK-DAG: ![[ELTTY16]] = !DIBasicType(name: "u16", size: 16, encoding: DW_ATE_unsigned)
    let u16: svuint16_t;

    // CHECK-DAG: name: "svint32_t",{{.*}}, baseType: ![[CT32:[0-9]+]]
    // CHECK-DAG: ![[CT32]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY32:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS32:[0-9]+]])
    // CHECK-DAG: ![[ELTTY32]] = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
    // CHECK-DAG: ![[ELTS32]] = !{![[REALELTS32:[0-9]+]]}
    // CHECK-DAG: ![[REALELTS32]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 2, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
    let s32: svint32_t;

    // CHECK-DAG: name: "svuint32_t",{{.*}}, baseType: ![[CT32:[0-9]+]]
    // CHECK-DAG: ![[CT32]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY32:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS32]])
    // CHECK-DAG: ![[ELTTY32]] = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
    let u32: svuint32_t;

    // CHECK-DAG: name: "svint64_t",{{.*}}, baseType: ![[CT64:[0-9]+]]
    // CHECK-DAG: ![[CT64]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY64:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS64:[0-9]+]])
    // CHECK-DAG: ![[ELTTY64]] = !DIBasicType(name: "i64", size: 64, encoding: DW_ATE_signed)
    // CHECK-DAG: ![[ELTS64]] = !{![[REALELTS64:[0-9]+]]}
    // CHECK-DAG: ![[REALELTS64]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 1, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
    let s64: svint64_t;

    // CHECK-DAG: name: "svuint64_t",{{.*}}, baseType: ![[CT64:[0-9]+]]
    // CHECK-DAG: ![[CT64]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY64:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS64]])
    // CHECK-DAG: ![[ELTTY64]] = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
    let u64: svuint64_t;

    // CHECK:     name: "svfloat32_t",{{.*}}, baseType: ![[CT32:[0-9]+]]
    // CHECK-DAG: ![[CT32]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY32:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS32]])
    // CHECK-DAG: ![[ELTTY32]] = !DIBasicType(name: "f32", size: 32, encoding: DW_ATE_float)
    let f32: svfloat32_t;

    // CHECK:     name: "svfloat64_t",{{.*}}, baseType: ![[CT64:[0-9]+]]
    // CHECK-DAG: ![[CT64]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY64:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS64]])
    // CHECK-DAG: ![[ELTTY64]] = !DIBasicType(name: "f64", size: 64, encoding: DW_ATE_float)
    let f64: svfloat64_t;
}
