//@ only-aarch64
//@ only-linux
//@ compile-flags: -Cdebuginfo=2 -Copt-level=0

#![crate_type = "lib"]
#![allow(incomplete_features, internal_features)]
#![feature(rustc_attrs)]

// Test that we generate the correct debuginfo for scalable vector types.

#[rustc_scalable_vector(16)]
#[allow(non_camel_case_types)]
struct svint8_t(i8);

#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
struct svint8x4_t(svint8_t, svint8_t, svint8_t, svint8_t);

#[rustc_scalable_vector(16)]
#[allow(non_camel_case_types)]
struct svuint8_t(u8);

#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
struct svuint8x4_t(svuint8_t, svuint8_t, svuint8_t, svuint8_t);

#[rustc_scalable_vector(8)]
#[allow(non_camel_case_types)]
struct svint16_t(i16);

#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
struct svint16x4_t(svint16_t, svint16_t, svint16_t, svint16_t);

#[rustc_scalable_vector(8)]
#[allow(non_camel_case_types)]
struct svuint16_t(u16);

#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
struct svuint16x4_t(svuint16_t, svuint16_t, svuint16_t, svuint16_t);

#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
struct svint32_t(i32);

#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
struct svint32x4_t(svint32_t, svint32_t, svint32_t, svint32_t);

#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
struct svuint32_t(u32);

#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
struct svuint32x4_t(svuint32_t, svuint32_t, svuint32_t, svuint32_t);

#[rustc_scalable_vector(2)]
#[allow(non_camel_case_types)]
struct svint64_t(i64);

#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
struct svint64x4_t(svint64_t, svint64_t, svint64_t, svint64_t);

#[rustc_scalable_vector(2)]
#[allow(non_camel_case_types)]
struct svuint64_t(u64);

#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
struct svuint64x4_t(svuint64_t, svuint64_t, svuint64_t, svuint64_t);

#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
struct svfloat32_t(f32);

#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
struct svfloat32x4_t(svfloat32_t, svfloat32_t, svfloat32_t, svfloat32_t);

#[rustc_scalable_vector(2)]
#[allow(non_camel_case_types)]
struct svfloat64_t(f64);

#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
struct svfloat64x4_t(svfloat64_t, svfloat64_t, svfloat64_t, svfloat64_t);

#[target_feature(enable = "sve")]
pub fn locals() {
    // CHECK-DAG: name: "svint8x4_t",{{.*}}, baseType: ![[CT8:[0-9]+]]
    // CHECK-DAG: ![[CT8]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY8:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS8x4:[0-9]+]])
    // CHECK-DAG: ![[ELTTY8]] = !DIBasicType(name: "i8", size: 8, encoding: DW_ATE_signed)
    // CHECK-DAG: ![[ELTS8x4]] = !{![[REALELTS8x4:[0-9]+]]}
    // CHECK-DAG: ![[REALELTS8x4]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 32, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
    let s8: svint8x4_t;

    // CHECK-DAG: name: "svuint8x4_t",{{.*}}, baseType: ![[CT8:[0-9]+]]
    // CHECK-DAG: ![[CT8]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY8:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS8x4]])
    // CHECK-DAG: ![[ELTTY8]] = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)
    let u8: svuint8x4_t;

    // CHECK-DAG: name: "svint16x4_t",{{.*}}, baseType: ![[CT16:[0-9]+]]
    // CHECK-DAG: ![[CT16]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY16:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS16x4:[0-9]+]])
    // CHECK-DAG: ![[ELTTY16]] = !DIBasicType(name: "i16", size: 16, encoding: DW_ATE_signed)
    // CHECK-DAG: ![[ELTS16x4]] = !{![[REALELTS16x4:[0-9]+]]}
    // CHECK-DAG: ![[REALELTS16x4]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 16, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
    let s16: svint16x4_t;

    // CHECK-DAG: name: "svuint16x4_t",{{.*}}, baseType: ![[CT16:[0-9]+]]
    // CHECK-DAG: ![[CT16]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY16:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS16x4]])
    // CHECK-DAG: ![[ELTTY16]] = !DIBasicType(name: "u16", size: 16, encoding: DW_ATE_unsigned)
    let u16: svuint16x4_t;

    // CHECK-DAG: name: "svint32x4_t",{{.*}}, baseType: ![[CT32:[0-9]+]]
    // CHECK-DAG: ![[CT32]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY32:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS32x4:[0-9]+]])
    // CHECK-DAG: ![[ELTTY32]] = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
    // CHECK-DAG: ![[ELTS32x4]] = !{![[REALELTS32x4:[0-9]+]]}
    // CHECK-DAG: ![[REALELTS32x4]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 8, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
    let s32: svint32x4_t;

    // CHECK-DAG: name: "svuint32x4_t",{{.*}}, baseType: ![[CT32:[0-9]+]]
    // CHECK-DAG: ![[CT32]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY32:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS32x4]])
    // CHECK-DAG: ![[ELTTY32]] = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
    let u32: svuint32x4_t;

    // CHECK-DAG: name: "svint64x4_t",{{.*}}, baseType: ![[CT64:[0-9]+]]
    // CHECK-DAG: ![[CT64]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY64:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS1x4_64:[0-9]+]])
    // CHECK-DAG: ![[ELTTY64]] = !DIBasicType(name: "i64", size: 64, encoding: DW_ATE_signed)
    // CHECK-DAG: ![[ELTS1x4_64]] = !{![[REALELTS1x4_64:[0-9]+]]}
    // CHECK-DAG: ![[REALELTS1x4_64]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 4, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
    let s64: svint64x4_t;

    // CHECK-DAG: name: "svuint64x4_t",{{.*}}, baseType: ![[CT64:[0-9]+]]
    // CHECK-DAG: ![[CT64]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY64:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS1x4_64]])
    // CHECK-DAG: ![[ELTTY64]] = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
    let u64: svuint64x4_t;

    // CHECK:     name: "svfloat32x4_t",{{.*}}, baseType: ![[CT32:[0-9]+]]
    // CHECK-DAG: ![[CT32]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY32:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS32x4]])
    // CHECK-DAG: ![[ELTTY32]] = !DIBasicType(name: "f32", size: 32, encoding: DW_ATE_float)
    let f32: svfloat32x4_t;

    // CHECK:     name: "svfloat64x4_t",{{.*}}, baseType: ![[CT64:[0-9]+]]
    // CHECK-DAG: ![[CT64]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY64:[0-9]+]],{{.*}}, flags: DIFlagVector, elements: ![[ELTS1x4_64]])
    // CHECK-DAG: ![[ELTTY64]] = !DIBasicType(name: "f64", size: 64, encoding: DW_ATE_float)
    let f64: svfloat64x4_t;
}
