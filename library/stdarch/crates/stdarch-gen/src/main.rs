use self::Suffix::*;
use self::TargetFeature::*;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::path::PathBuf;

const IN: &str = "neon.spec";
const ARM_OUT: &str = "generated.rs";
const AARCH64_OUT: &str = "generated.rs";

const UINT_TYPES: [&str; 6] = [
    "uint8x8_t",
    "uint8x16_t",
    "uint16x4_t",
    "uint16x8_t",
    "uint32x2_t",
    "uint32x4_t",
];

const UINT_TYPES_64: [&str; 2] = ["uint64x1_t", "uint64x2_t"];

const INT_TYPES: [&str; 6] = [
    "int8x8_t",
    "int8x16_t",
    "int16x4_t",
    "int16x8_t",
    "int32x2_t",
    "int32x4_t",
];

const INT_TYPES_64: [&str; 2] = ["int64x1_t", "int64x2_t"];

const FLOAT_TYPES: [&str; 2] = [
    //"float8x8_t", not supported by rust
    //"float8x16_t", not supported by rust
    //"float16x4_t", not supported by rust
    //"float16x8_t", not supported by rust
    "float32x2_t",
    "float32x4_t",
];

const FLOAT_TYPES_64: [&str; 2] = [
    //"float8x8_t", not supported by rust
    //"float8x16_t", not supported by rust
    //"float16x4_t", not supported by rust
    //"float16x8_t", not supported by rust
    "float64x1_t",
    "float64x2_t",
];

fn type_len(t: &str) -> usize {
    let s: Vec<_> = t.split("x").collect();
    if s.len() == 2 {
        match &s[1][0..2] {
            "1_" => 1,
            "2_" => 2,
            "4_" => 4,
            "8_" => 8,
            "16" => 16,
            _ => panic!("unknown type: {}", t),
        }
    } else if s.len() == 3 {
        s[1].parse::<usize>().unwrap() * type_sub_len(t)
    } else {
        1
    }
}

fn type_sub_len(t: &str) -> usize {
    let s: Vec<_> = t.split('x').collect();
    if s.len() != 3 {
        1
    } else {
        match s[2] {
            "2_t" => 2,
            "3_t" => 3,
            "4_t" => 4,
            _ => panic!("unknown type len: {}", t),
        }
    }
}

fn type_bits(t: &str) -> usize {
    match t {
        "int8x8_t" | "int8x16_t" | "uint8x8_t" | "uint8x16_t" | "poly8x8_t" | "poly8x16_t"
        | "i8" | "u8" => 8,
        "int16x4_t" | "int16x8_t" | "uint16x4_t" | "uint16x8_t" | "poly16x4_t" | "poly16x8_t"
        | "i16" | "u16" => 16,
        "int32x2_t" | "int32x4_t" | "uint32x2_t" | "uint32x4_t" | "i32" | "u32" | "float32x2_t"
        | "float32x4_t" | "f32" => 32,
        "int64x1_t" | "int64x2_t" | "uint64x1_t" | "uint64x2_t" | "poly64x1_t" | "poly64x2_t"
        | "i64" | "u64" | "float64x1_t" | "float64x2_t" | "f64" => 64,
        _ => panic!("unknown type: {}", t),
    }
}

fn type_exp_len(t: &str, base_len: usize) -> usize {
    let t = type_to_sub_type(t);
    let len = type_len(&t) / base_len;
    match len {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        16 => 4,
        _ => panic!("unknown type: {}", t),
    }
}

fn type_bits_exp_len(t: &str) -> usize {
    match t {
        "int8x8_t" | "int8x16_t" | "uint8x8_t" | "uint8x16_t" | "poly8x8_t" | "poly8x16_t"
        | "i8" | "u8" => 3,
        "int16x4_t" | "int16x8_t" | "uint16x4_t" | "uint16x8_t" | "poly16x4_t" | "poly16x8_t"
        | "i16" | "u16" => 4,
        "int32x2_t" | "int32x4_t" | "uint32x2_t" | "uint32x4_t" | "i32" | "u32" => 5,
        "int64x1_t" | "int64x2_t" | "uint64x1_t" | "uint64x2_t" | "poly64x1_t" | "poly64x2_t"
        | "i64" | "u64" => 6,
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_suffix(t: &str) -> &str {
    match t {
        "int8x8_t" => "_s8",
        "int8x16_t" => "q_s8",
        "int16x4_t" => "_s16",
        "int16x8_t" => "q_s16",
        "int32x2_t" => "_s32",
        "int32x4_t" => "q_s32",
        "int64x1_t" => "_s64",
        "int64x2_t" => "q_s64",
        "uint8x8_t" => "_u8",
        "uint8x16_t" => "q_u8",
        "uint16x4_t" => "_u16",
        "uint16x8_t" => "q_u16",
        "uint32x2_t" => "_u32",
        "uint32x4_t" => "q_u32",
        "uint64x1_t" => "_u64",
        "uint64x2_t" => "q_u64",
        "float16x4_t" => "_f16",
        "float16x8_t" => "q_f16",
        "float32x2_t" => "_f32",
        "float32x4_t" => "q_f32",
        "float64x1_t" => "_f64",
        "float64x2_t" => "q_f64",
        "poly8x8_t" => "_p8",
        "poly8x16_t" => "q_p8",
        "poly16x4_t" => "_p16",
        "poly16x8_t" => "q_p16",
        "poly64x1_t" => "_p64",
        "poly64x2_t" => "q_p64",
        "int8x8x2_t" => "_s8_x2",
        "int8x8x3_t" => "_s8_x3",
        "int8x8x4_t" => "_s8_x4",
        "int16x4x2_t" => "_s16_x2",
        "int16x4x3_t" => "_s16_x3",
        "int16x4x4_t" => "_s16_x4",
        "int32x2x2_t" => "_s32_x2",
        "int32x2x3_t" => "_s32_x3",
        "int32x2x4_t" => "_s32_x4",
        "int64x1x2_t" => "_s64_x2",
        "int64x1x3_t" => "_s64_x3",
        "int64x1x4_t" => "_s64_x4",
        "uint8x8x2_t" => "_u8_x2",
        "uint8x8x3_t" => "_u8_x3",
        "uint8x8x4_t" => "_u8_x4",
        "uint16x4x2_t" => "_u16_x2",
        "uint16x4x3_t" => "_u16_x3",
        "uint16x4x4_t" => "_u16_x4",
        "uint32x2x2_t" => "_u32_x2",
        "uint32x2x3_t" => "_u32_x3",
        "uint32x2x4_t" => "_u32_x4",
        "uint64x1x2_t" => "_u64_x2",
        "uint64x1x3_t" => "_u64_x3",
        "uint64x1x4_t" => "_u64_x4",
        "poly8x8x2_t" => "_p8_x2",
        "poly8x8x3_t" => "_p8_x3",
        "poly8x8x4_t" => "_p8_x4",
        "poly16x4x2_t" => "_p16_x2",
        "poly16x4x3_t" => "_p16_x3",
        "poly16x4x4_t" => "_p16_x4",
        "poly64x1x2_t" => "_p64_x2",
        "poly64x1x3_t" => "_p64_x3",
        "poly64x1x4_t" => "_p64_x4",
        "float32x2x2_t" => "_f32_x2",
        "float32x2x3_t" => "_f32_x3",
        "float32x2x4_t" => "_f32_x4",
        "float64x1x2_t" => "_f64_x2",
        "float64x1x3_t" => "_f64_x3",
        "float64x1x4_t" => "_f64_x4",
        "int8x16x2_t" => "q_s8_x2",
        "int8x16x3_t" => "q_s8_x3",
        "int8x16x4_t" => "q_s8_x4",
        "int16x8x2_t" => "q_s16_x2",
        "int16x8x3_t" => "q_s16_x3",
        "int16x8x4_t" => "q_s16_x4",
        "int32x4x2_t" => "q_s32_x2",
        "int32x4x3_t" => "q_s32_x3",
        "int32x4x4_t" => "q_s32_x4",
        "int64x2x2_t" => "q_s64_x2",
        "int64x2x3_t" => "q_s64_x3",
        "int64x2x4_t" => "q_s64_x4",
        "uint8x16x2_t" => "q_u8_x2",
        "uint8x16x3_t" => "q_u8_x3",
        "uint8x16x4_t" => "q_u8_x4",
        "uint16x8x2_t" => "q_u16_x2",
        "uint16x8x3_t" => "q_u16_x3",
        "uint16x8x4_t" => "q_u16_x4",
        "uint32x4x2_t" => "q_u32_x2",
        "uint32x4x3_t" => "q_u32_x3",
        "uint32x4x4_t" => "q_u32_x4",
        "uint64x2x2_t" => "q_u64_x2",
        "uint64x2x3_t" => "q_u64_x3",
        "uint64x2x4_t" => "q_u64_x4",
        "poly8x16x2_t" => "q_p8_x2",
        "poly8x16x3_t" => "q_p8_x3",
        "poly8x16x4_t" => "q_p8_x4",
        "poly16x8x2_t" => "q_p16_x2",
        "poly16x8x3_t" => "q_p16_x3",
        "poly16x8x4_t" => "q_p16_x4",
        "poly64x2x2_t" => "q_p64_x2",
        "poly64x2x3_t" => "q_p64_x3",
        "poly64x2x4_t" => "q_p64_x4",
        "float32x4x2_t" => "q_f32_x2",
        "float32x4x3_t" => "q_f32_x3",
        "float32x4x4_t" => "q_f32_x4",
        "float64x2x2_t" => "q_f64_x2",
        "float64x2x3_t" => "q_f64_x3",
        "float64x2x4_t" => "q_f64_x4",
        "i8" => "b_s8",
        "i16" => "h_s16",
        "i32" => "s_s32",
        "i64" => "d_s64",
        "u8" => "b_u8",
        "u16" => "h_u16",
        "u32" => "s_u32",
        "u64" => "d_u64",
        "f32" => "s_f32",
        "f64" => "d_f64",
        "p8" => "b_p8",
        "p16" => "h_p16",
        "p128" => "q_p128",
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_dup_suffix(t: &str) -> String {
    let s: Vec<_> = type_to_suffix(t).split('_').collect();
    assert_eq!(s.len(), 2);
    format!("{}_dup_{}", s[0], s[1])
}

fn type_to_lane_suffix(t: &str) -> String {
    let s: Vec<_> = type_to_suffix(t).split('_').collect();
    assert_eq!(s.len(), 2);
    format!("{}_lane_{}", s[0], s[1])
}

fn type_to_n_suffix(t: &str) -> &str {
    match t {
        "int8x8_t" => "_n_s8",
        "int8x16_t" => "q_n_s8",
        "int16x4_t" => "_n_s16",
        "int16x8_t" => "q_n_s16",
        "int32x2_t" => "_n_s32",
        "int32x4_t" => "q_n_s32",
        "int64x1_t" => "_n_s64",
        "int64x2_t" => "q_n_s64",
        "uint8x8_t" => "_n_u8",
        "uint8x16_t" => "q_n_u8",
        "uint16x4_t" => "_n_u16",
        "uint16x8_t" => "q_n_u16",
        "uint32x2_t" => "_n_u32",
        "uint32x4_t" => "q_n_u32",
        "uint64x1_t" => "_n_u64",
        "uint64x2_t" => "q_n_u64",
        "float16x4_t" => "_n_f16",
        "float16x8_t" => "q_n_f16",
        "float32x2_t" => "_n_f32",
        "float32x4_t" => "q_n_f32",
        "float64x1_t" => "_n_f64",
        "float64x2_t" => "q_n_f64",
        "poly8x8_t" => "_n_p8",
        "poly8x16_t" => "q_n_p8",
        "poly16x4_t" => "_n_p16",
        "poly16x8_t" => "q_n_p16",
        "poly64x1_t" => "_n_p64",
        "poly64x2_t" => "q_n_p64",
        "i8" => "b_n_s8",
        "i16" => "h_n_s16",
        "i32" => "s_n_s32",
        "i64" => "d_n_s64",
        "u8" => "b_n_u8",
        "u16" => "h_n_u16",
        "u32" => "s_n_u32",
        "u64" => "d_n_u64",
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_noq_n_suffix(t: &str) -> &str {
    match t {
        "int8x8_t" | "int8x16_t" => "_n_s8",
        "int16x4_t" | "int16x8_t" => "_n_s16",
        "int32x2_t" | "int32x4_t" => "_n_s32",
        "int64x1_t" | "int64x2_t" => "_n_s64",
        "uint8x8_t" | "uint8x16_t" => "_n_u8",
        "uint16x4_t" | "uint16x8_t" => "_n_u16",
        "uint32x2_t" | "uint32x4_t" => "_n_u32",
        "uint64x1_t" | "uint64x2_t" => "_n_u64",
        "float16x4_t" | "float16x8_t" => "_n_f16",
        "float32x2_t" | "float32x4_t" => "_n_f32",
        "float64x1_t" | "float64x2_t" => "_n_f64",
        "poly8x8_t" | "poly8x16_t" => "_n_p8",
        "poly16x4_t" | "poly16x8_t" => "_n_p16",
        "poly64x1_t" | "poly64x2_t" => "_n_p64",
        "i8" => "b_n_s8",
        "i16" => "h_n_s16",
        "i32" => "s_n_s32",
        "i64" => "d_n_s64",
        "u8" => "b_n_u8",
        "u16" => "h_n_u16",
        "u32" => "s_n_u32",
        "u64" => "d_n_u64",
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_lane_suffixes<'a>(out_t: &'a str, in_t: &'a str, re_to_out: bool) -> String {
    let mut str = String::new();
    let suf = type_to_suffix(out_t);
    if !suf.starts_with("_") {
        str.push_str(&suf[0..1]);
    }
    str.push_str("_lane");
    if !re_to_out {
        str.push_str(type_to_suffix(in_t));
    } else {
        if type_to_suffix(in_t).starts_with("q") {
            str.push_str("q");
        };
        let suf2 = type_to_noq_suffix(out_t);
        str.push_str(suf2);
    }
    str
}

fn type_to_rot_suffix(c_name: &str, suf: &str) -> String {
    let ns: Vec<_> = c_name.split('_').collect();
    assert_eq!(ns.len(), 2);
    if suf.starts_with("q") {
        format!("{}q_{}{}", ns[0], ns[1], &suf[1..])
    } else {
        format!("{}{}", c_name, suf)
    }
}

fn type_to_signed(t: &str) -> String {
    let s = t.replace("uint", "int");
    let s = s.replace("poly", "int");
    s
}

fn type_to_unsigned(t: &str) -> String {
    if t.contains("uint") {
        return t.to_string();
    }
    let s = t.replace("int", "uint");
    let s = s.replace("poly", "uint");
    s
}

fn type_to_double_suffixes<'a>(out_t: &'a str, in_t: &'a str) -> String {
    let mut str = String::new();
    let suf = type_to_suffix(in_t);
    if suf.starts_with("q") && type_to_suffix(out_t).starts_with("q") {
        str.push_str("q");
    }
    if !suf.starts_with("_") && !suf.starts_with("q") {
        str.push_str(&suf[0..1]);
    }
    str.push_str(type_to_noq_suffix(out_t));
    str.push_str(type_to_noq_suffix(in_t));
    str
}

fn type_to_double_n_suffixes<'a>(out_t: &'a str, in_t: &'a str) -> String {
    let mut str = String::new();
    let suf = type_to_suffix(in_t);
    if suf.starts_with("q") && type_to_suffix(out_t).starts_with("q") {
        str.push_str("q");
    }
    if !suf.starts_with("_") && !suf.starts_with("q") {
        str.push_str(&suf[0..1]);
    }
    str.push_str("_n");
    str.push_str(type_to_noq_suffix(out_t));
    str.push_str(type_to_noq_suffix(in_t));
    str
}

fn type_to_noq_double_suffixes<'a>(out_t: &'a str, in_t: &'a str) -> String {
    let mut str = String::new();
    str.push_str(type_to_noq_suffix(out_t));
    str.push_str(type_to_noq_suffix(in_t));
    str
}

fn type_to_noq_suffix(t: &str) -> &str {
    match t {
        "int8x8_t" | "int8x16_t" | "i8" => "_s8",
        "int16x4_t" | "int16x8_t" | "i16" => "_s16",
        "int32x2_t" | "int32x4_t" | "i32" => "_s32",
        "int64x1_t" | "int64x2_t" | "i64" => "_s64",
        "uint8x8_t" | "uint8x16_t" | "u8" => "_u8",
        "uint16x4_t" | "uint16x8_t" | "u16" => "_u16",
        "uint32x2_t" | "uint32x4_t" | "u32" => "_u32",
        "uint64x1_t" | "uint64x2_t" | "u64" => "_u64",
        "float16x4_t" | "float16x8_t" => "_f16",
        "float32x2_t" | "float32x4_t" | "f32" => "_f32",
        "float64x1_t" | "float64x2_t" | "f64" => "_f64",
        "poly8x8_t" | "poly8x16_t" => "_p8",
        "poly16x4_t" | "poly16x8_t" => "_p16",
        "poly64x1_t" | "poly64x2_t" | "p64" => "_p64",
        "p128" => "_p128",
        _ => panic!("unknown type: {}", t),
    }
}

#[derive(Clone, Copy)]
enum Suffix {
    Normal,
    Double,
    NoQ,
    NoQDouble,
    NSuffix,
    DoubleN,
    NoQNSuffix,
    OutSuffix,
    OutNSuffix,
    OutNox,
    In1Nox,
    OutDupNox,
    OutLaneNox,
    In1LaneNox,
    Lane,
    In2,
    In2Lane,
    OutLane,
    Rot,
    RotLane,
}

#[derive(Clone, Copy)]
enum TargetFeature {
    Default,
    ArmV7,
    Vfp4,
    FPArmV8,
    AES,
    FCMA,
    Dotprod,
    I8MM,
    SHA3,
    RDM,
    SM4,
    FTTS,
}

#[derive(Clone, Copy)]
enum Fntype {
    Normal,
    Load,
    Store,
}

fn type_to_global_type(t: &str) -> &str {
    match t {
        "int8x8_t" | "int8x8x2_t" | "int8x8x3_t" | "int8x8x4_t" => "i8x8",
        "int8x16_t" | "int8x16x2_t" | "int8x16x3_t" | "int8x16x4_t" => "i8x16",
        "int16x4_t" | "int16x4x2_t" | "int16x4x3_t" | "int16x4x4_t" => "i16x4",
        "int16x8_t" | "int16x8x2_t" | "int16x8x3_t" | "int16x8x4_t" => "i16x8",
        "int32x2_t" | "int32x2x2_t" | "int32x2x3_t" | "int32x2x4_t" => "i32x2",
        "int32x4_t" | "int32x4x2_t" | "int32x4x3_t" | "int32x4x4_t" => "i32x4",
        "int64x1_t" | "int64x1x2_t" | "int64x1x3_t" | "int64x1x4_t" => "i64x1",
        "int64x2_t" | "int64x2x2_t" | "int64x2x3_t" | "int64x2x4_t" => "i64x2",
        "uint8x8_t" | "uint8x8x2_t" | "uint8x8x3_t" | "uint8x8x4_t" => "u8x8",
        "uint8x16_t" | "uint8x16x2_t" | "uint8x16x3_t" | "uint8x16x4_t" => "u8x16",
        "uint16x4_t" | "uint16x4x2_t" | "uint16x4x3_t" | "uint16x4x4_t" => "u16x4",
        "uint16x8_t" | "uint16x8x2_t" | "uint16x8x3_t" | "uint16x8x4_t" => "u16x8",
        "uint32x2_t" | "uint32x2x2_t" | "uint32x2x3_t" | "uint32x2x4_t" => "u32x2",
        "uint32x4_t" | "uint32x4x2_t" | "uint32x4x3_t" | "uint32x4x4_t" => "u32x4",
        "uint64x1_t" | "uint64x1x2_t" | "uint64x1x3_t" | "uint64x1x4_t" => "u64x1",
        "uint64x2_t" | "uint64x2x2_t" | "uint64x2x3_t" | "uint64x2x4_t" => "u64x2",
        "float16x4_t" => "f16x4",
        "float16x8_t" => "f16x8",
        "float32x2_t" | "float32x2x2_t" | "float32x2x3_t" | "float32x2x4_t" => "f32x2",
        "float32x4_t" | "float32x4x2_t" | "float32x4x3_t" | "float32x4x4_t" => "f32x4",
        "float64x1_t" | "float64x1x2_t" | "float64x1x3_t" | "float64x1x4_t" => "f64",
        "float64x2_t" | "float64x2x2_t" | "float64x2x3_t" | "float64x2x4_t" => "f64x2",
        "poly8x8_t" | "poly8x8x2_t" | "poly8x8x3_t" | "poly8x8x4_t" => "i8x8",
        "poly8x16_t" | "poly8x16x2_t" | "poly8x16x3_t" | "poly8x16x4_t" => "i8x16",
        "poly16x4_t" | "poly16x4x2_t" | "poly16x4x3_t" | "poly16x4x4_t" => "i16x4",
        "poly16x8_t" | "poly16x8x2_t" | "poly16x8x3_t" | "poly16x8x4_t" => "i16x8",
        "poly64x1_t" | "poly64x1x2_t" | "poly64x1x3_t" | "poly64x1x4_t" => "i64x1",
        "poly64x2_t" | "poly64x2x2_t" | "poly64x2x3_t" | "poly64x2x4_t" => "i64x2",
        "i8" => "i8",
        "i16" => "i16",
        "i32" => "i32",
        "i64" => "i64",
        "u8" => "u8",
        "u16" => "u16",
        "u32" => "u32",
        "u64" => "u64",
        "f32" => "f32",
        "f64" => "f64",
        "p8" => "p8",
        "p16" => "p16",
        "p64" => "p64",
        "p128" => "p128",
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_sub_type(t: &str) -> String {
    let s: Vec<_> = t.split('x').collect();
    match s.len() {
        2 => String::from(t),
        3 => format!("{}x{}_t", s[0], s[1]),
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_native_type(t: &str) -> String {
    let s: Vec<_> = t.split('x').collect();
    match s.len() {
        1 => {
            assert!(t.contains("*const") || t.contains("*mut"));
            let sub: Vec<_> = t.split(' ').collect();
            String::from(sub[1])
        }
        2 | 3 => match &s[0][0..3] {
            "int" => format!("i{}", &s[0][3..]),
            "uin" => format!("u{}", &s[0][4..]),
            "flo" => format!("f{}", &s[0][5..]),
            "pol" => format!("u{}", &s[0][4..]),
            _ => panic!("unknown type: {}", t),
        },
        _ => panic!("unknown type: {}", t),
    }
}

fn native_type_to_type(t: &str) -> &str {
    match t {
        "i8" => "int8x8_t",
        "i16" => "int16x4_t",
        "i32" => "int32x2_t",
        "i64" => "int64x1_t",
        "u8" => "uint8x8_t",
        "u16" => "uint16x4_t",
        "u32" => "uint32x2_t",
        "u64" => "uint64x1_t",
        "f16" => "float16x4_t",
        "f32" => "float32x2_t",
        "f64" => "float64x1_t",
        _ => panic!("unknown type: {}", t),
    }
}

fn native_type_to_long_type(t: &str) -> &str {
    match t {
        "i8" => "int8x16_t",
        "i16" => "int16x8_t",
        "i32" => "int32x4_t",
        "i64" => "int64x2_t",
        "u8" => "uint8x16_t",
        "u16" => "uint16x8_t",
        "u32" => "uint32x4_t",
        "u64" => "uint64x2_t",
        "f16" => "float16x8_t",
        "f32" => "float32x4_t",
        "f64" => "float64x2_t",
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_half(t: &str) -> &str {
    match t {
        "int8x16_t" => "int8x8_t",
        "int16x8_t" => "int16x4_t",
        "int32x4_t" => "int32x2_t",
        "int64x2_t" => "int64x1_t",
        "uint8x16_t" => "uint8x8_t",
        "uint16x8_t" => "uint16x4_t",
        "uint32x4_t" => "uint32x2_t",
        "uint64x2_t" => "uint64x1_t",
        "poly8x16_t" => "poly8x8_t",
        "poly16x8_t" => "poly16x4_t",
        "float32x4_t" => "float32x2_t",
        "float64x2_t" => "float64x1_t",
        _ => panic!("unknown half type for {}", t),
    }
}

fn asc(start: i32, len: usize) -> String {
    let mut s = String::from("[");
    for i in 0..len {
        if i != 0 {
            s.push_str(", ");
        }
        let n = start + i as i32;
        s.push_str(&n.to_string());
    }
    s.push_str("]");
    s
}

fn transpose1(x: usize) -> &'static str {
    match x {
        2 => "[0, 2]",
        4 => "[0, 4, 2, 6]",
        8 => "[0, 8, 2, 10, 4, 12, 6, 14]",
        16 => "[0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30]",
        _ => panic!("unknown transpose order of len {}", x),
    }
}

fn transpose2(x: usize) -> &'static str {
    match x {
        2 => "[1, 3]",
        4 => "[1, 5, 3, 7]",
        8 => "[1, 9, 3, 11, 5, 13, 7, 15]",
        16 => "[1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31]",
        _ => panic!("unknown transpose order of len {}", x),
    }
}

fn zip1(x: usize) -> &'static str {
    match x {
        2 => "[0, 2]",
        4 => "[0, 4, 1, 5]",
        8 => "[0, 8, 1, 9, 2, 10, 3, 11]",
        16 => "[0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]",
        _ => panic!("unknown zip order of len {}", x),
    }
}

fn zip2(x: usize) -> &'static str {
    match x {
        2 => "[1, 3]",
        4 => "[2, 6, 3, 7]",
        8 => "[4, 12, 5, 13, 6, 14, 7, 15]",
        16 => "[8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]",
        _ => panic!("unknown zip order of len {}", x),
    }
}

fn unzip1(x: usize) -> &'static str {
    match x {
        2 => "[0, 2]",
        4 => "[0, 2, 4, 6]",
        8 => "[0, 2, 4, 6, 8, 10, 12, 14]",
        16 => "[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]",
        _ => panic!("unknown unzip order of len {}", x),
    }
}

fn unzip2(x: usize) -> &'static str {
    match x {
        2 => "[1, 3]",
        4 => "[1, 3, 5, 7]",
        8 => "[1, 3, 5, 7, 9, 11, 13, 15]",
        16 => "[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]",
        _ => panic!("unknown unzip order of len {}", x),
    }
}

fn values(t: &str, vs: &[String]) -> String {
    if vs.len() == 1 && !t.contains('x') {
        format!(": {} = {}", t, vs[0])
    } else if vs.len() == 1 && type_to_global_type(t) == "f64" {
        format!(": {} = {}", type_to_global_type(t), vs[0])
    } else {
        let s: Vec<_> = t.split('x').collect();
        if s.len() == 3 {
            format!(
                ": [{}; {}] = [{}]",
                type_to_native_type(t),
                type_len(t),
                vs.iter()
                    .map(|v| map_val(type_to_global_type(t), v))
                    //.map(|v| format!("{}{}", v, type_to_native_type(t)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            format!(
                ": {} = {}::new({})",
                type_to_global_type(t),
                type_to_global_type(t),
                vs.iter()
                    .map(|v| map_val(type_to_global_type(t), v))
                    //.map(|v| format!("{}{}", v, type_to_native_type(t)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

fn max_val(t: &str) -> &'static str {
    match &t[..3] {
        "u8x" => "0xFF",
        "u16" => "0xFF_FF",
        "u32" => "0xFF_FF_FF_FF",
        "u64" => "0xFF_FF_FF_FF_FF_FF_FF_FF",
        "i8x" => "0x7F",
        "i16" => "0x7F_FF",
        "i32" => "0x7F_FF_FF_FF",
        "i64" => "0x7F_FF_FF_FF_FF_FF_FF_FF",
        "f32" => "3.40282347e+38",
        "f64" => "1.7976931348623157e+308",
        _ => panic!("No TRUE for type {}", t),
    }
}

fn min_val(t: &str) -> &'static str {
    match &t[..3] {
        "u8x" => "0",
        "u16" => "0",
        "u32" => "0",
        "u64" => "0",
        "i8x" => "-128",
        "i16" => "-32768",
        "i32" => "-2147483648",
        "i64" => "-9223372036854775808",
        "f32" => "-3.40282347e+38",
        "f64" => "-1.7976931348623157e+308",
        _ => panic!("No TRUE for type {}", t),
    }
}

fn true_val(t: &str) -> &'static str {
    match &t[..3] {
        "u8x" => "0xFF",
        "u16" => "0xFF_FF",
        "u32" => "0xFF_FF_FF_FF",
        "u64" => "0xFF_FF_FF_FF_FF_FF_FF_FF",
        _ => panic!("No TRUE for type {}", t),
    }
}

fn ff_val(t: &str) -> &'static str {
    match &t[..3] {
        "u8x" => "0xFF",
        "u16" => "0xFF_FF",
        "u32" => "0xFF_FF_FF_FF",
        "u64" => "0xFF_FF_FF_FF_FF_FF_FF_FF",
        "i8x" => "0xFF",
        "i16" => "0xFF_FF",
        "i32" => "0xFF_FF_FF_FF",
        "i64" => "0xFF_FF_FF_FF_FF_FF_FF_FF",
        _ => panic!("No TRUE for type {}", t),
    }
}

fn false_val(_t: &str) -> &'static str {
    "0"
}

fn bits(t: &str) -> &'static str {
    match &t[..3] {
        "u8x" => "8",
        "u16" => "16",
        "u32" => "32",
        "u64" => "64",
        "i8x" => "8",
        "i16" => "16",
        "i32" => "32",
        "i64" => "64",
        "p8x" => "8",
        "p16" => "16",
        "p64" => "64",
        _ => panic!("Unknown bits for type {}", t),
    }
}

fn bits_minus_one(t: &str) -> &'static str {
    match &t[..3] {
        "u8x" => "7",
        "u16" => "15",
        "u32" => "31",
        "u64" => "63",
        "i8x" => "7",
        "i16" => "15",
        "i32" => "31",
        "i64" => "63",
        "p8x" => "7",
        "p16" => "15",
        "p64" => "63",
        _ => panic!("Unknown bits for type {}", t),
    }
}

fn half_bits(t: &str) -> &'static str {
    match &t[..3] {
        "u8x" => "4",
        "u16" => "8",
        "u32" => "16",
        "u64" => "32",
        "i8x" => "4",
        "i16" => "8",
        "i32" => "16",
        "i64" => "32",
        "p8x" => "4",
        "p16" => "8",
        "p64" => "32",
        _ => panic!("Unknown bits for type {}", t),
    }
}

fn type_len_str(t: &str) -> &'static str {
    match t {
        "int8x8_t" => "8",
        "int8x16_t" => "16",
        "int16x4_t" => "4",
        "int16x8_t" => "8",
        "int32x2_t" => "2",
        "int32x4_t" => "4",
        "int64x1_t" => "1",
        "int64x2_t" => "2",
        "uint8x8_t" => "8",
        "uint8x16_t" => "16",
        "uint16x4_t" => "4",
        "uint16x8_t" => "8",
        "uint32x2_t" => "2",
        "uint32x4_t" => "4",
        "uint64x1_t" => "1",
        "uint64x2_t" => "2",
        "float16x4_t" => "4",
        "float16x8_t" => "8",
        "float32x2_t" => "2",
        "float32x4_t" => "4",
        "float64x1_t" => "1",
        "float64x2_t" => "2",
        "poly8x8_t" => "8",
        "poly8x16_t" => "16",
        "poly16x4_t" => "4",
        "poly16x8_t" => "8",
        "poly64x1_t" => "1",
        "poly64x2_t" => "2",
        _ => panic!("unknown type: {}", t),
    }
}

fn type_half_len_str(t: &str) -> &'static str {
    match t {
        "int8x8_t" => "4",
        "int8x16_t" => "8",
        "int16x4_t" => "2",
        "int16x8_t" => "4",
        "int32x2_t" => "1",
        "int32x4_t" => "2",
        "int64x1_t" => "0",
        "int64x2_t" => "1",
        "uint8x8_t" => "4",
        "uint8x16_t" => "8",
        "uint16x4_t" => "2",
        "uint16x8_t" => "4",
        "uint32x2_t" => "1",
        "uint32x4_t" => "2",
        "uint64x1_t" => "0",
        "uint64x2_t" => "1",
        "float16x4_t" => "2",
        "float16x8_t" => "4",
        "float32x2_t" => "1",
        "float32x4_t" => "2",
        "float64x1_t" => "0",
        "float64x2_t" => "1",
        "poly8x8_t" => "4",
        "poly8x16_t" => "8",
        "poly16x4_t" => "2",
        "poly16x8_t" => "4",
        "poly64x1_t" => "0",
        "poly64x2_t" => "1",
        _ => panic!("unknown type: {}", t),
    }
}

fn map_val<'v>(t: &str, v: &'v str) -> &'v str {
    match v {
        "FALSE" => false_val(t),
        "TRUE" => true_val(t),
        "MAX" => max_val(t),
        "MIN" => min_val(t),
        "FF" => ff_val(t),
        "BITS" => bits(t),
        "BITS_M1" => bits_minus_one(t),
        "HFBITS" => half_bits(t),
        "LEN" => type_len_str(t),
        "HFLEN" => type_half_len_str(t),
        o => o,
    }
}

fn type_to_ext(t: &str, v: bool, r: bool, pi8: bool) -> String {
    if !t.contains('x') {
        return t.replace("u", "i");
    }
    let native = type_to_native_type(t);
    let sub_ext = match type_sub_len(t) {
        1 => String::new(),
        _ if v => format!(
            ".p0v{}{}",
            &type_len(&type_to_sub_type(t)).to_string(),
            native
        ),
        _ if pi8 => format!(".p0i8"),
        _ => format!(".p0{}", native),
    };
    let sub_type = match &native[0..1] {
        "i" | "f" => native,
        "u" => native.replace("u", "i"),
        _ => panic!("unknown type: {}", t),
    };
    let ext = format!(
        "v{}{}{}",
        &type_len(&type_to_sub_type(t)).to_string(),
        sub_type,
        sub_ext
    );
    if r {
        let ss: Vec<_> = ext.split('.').collect();
        if ss.len() != 2 {
            ext
        } else {
            format!("{}.{}", ss[1], ss[0])
        }
    } else {
        ext
    }
}

fn ext(s: &str, in_t: &[&str; 3], out_t: &str) -> String {
    s.replace("_EXT_", &type_to_ext(in_t[0], false, false, false))
        .replace("_EXT2_", &type_to_ext(out_t, false, false, false))
        .replace("_EXT3_", &type_to_ext(in_t[1], false, false, false))
        .replace("_EXT4_", &type_to_ext(in_t[2], false, false, false))
        .replace("_EXTr3_", &type_to_ext(in_t[1], false, true, false))
        .replace("_EXTv2_", &type_to_ext(out_t, true, false, false))
        .replace("_EXTpi8_", &type_to_ext(in_t[1], false, false, true))
        .replace("_EXTpi82_", &type_to_ext(out_t, false, false, true))
        .replace("_EXTpi8r_", &type_to_ext(in_t[1], false, true, true))
}

fn is_vldx(name: &str) -> bool {
    let s: Vec<_> = name.split('_').collect();
    &name[0..3] == "vld"
        && name[3..4].parse::<i32>().unwrap() > 1
        && (s.last().unwrap().starts_with("s") || s.last().unwrap().starts_with("f"))
}

fn is_vstx(name: &str) -> bool {
    let s: Vec<_> = name.split('_').collect();
    s.len() == 2
        && &name[0..3] == "vst"
        && name[3..4].parse::<i32>().unwrap() > 1
        && (s[1].starts_with("s") || s[1].starts_with("f"))
}

#[allow(clippy::too_many_arguments)]
fn gen_aarch64(
    current_comment: &str,
    current_fn: &Option<String>,
    current_name: &str,
    current_aarch64: &Option<String>,
    link_aarch64: &Option<String>,
    const_aarch64: &Option<String>,
    constn: &Option<String>,
    in_t: &[&str; 3],
    out_t: &str,
    current_tests: &[(
        Vec<String>,
        Vec<String>,
        Vec<String>,
        Option<String>,
        Vec<String>,
    )],
    suffix: Suffix,
    para_num: i32,
    target: TargetFeature,
    fixed: &Vec<String>,
    multi_fn: &Vec<String>,
    fn_type: Fntype,
) -> (String, String) {
    let name = match suffix {
        Normal => format!("{}{}", current_name, type_to_suffix(in_t[1])),
        NoQ => format!("{}{}", current_name, type_to_noq_suffix(in_t[1])),
        Double => format!(
            "{}{}",
            current_name,
            type_to_double_suffixes(out_t, in_t[1])
        ),
        NoQDouble => format!(
            "{}{}",
            current_name,
            type_to_noq_double_suffixes(out_t, in_t[1])
        ),
        NSuffix => format!("{}{}", current_name, type_to_n_suffix(in_t[1])),
        DoubleN => format!(
            "{}{}",
            current_name,
            type_to_double_n_suffixes(out_t, in_t[1])
        ),
        NoQNSuffix => format!("{}{}", current_name, type_to_noq_n_suffix(in_t[1])),
        OutSuffix => format!("{}{}", current_name, type_to_suffix(out_t)),
        OutNSuffix => format!("{}{}", current_name, type_to_n_suffix(out_t)),
        OutNox => format!(
            "{}{}",
            current_name,
            type_to_suffix(&type_to_sub_type(out_t))
        ),
        In1Nox => format!(
            "{}{}",
            current_name,
            type_to_suffix(&type_to_sub_type(in_t[1]))
        ),
        OutDupNox => format!(
            "{}{}",
            current_name,
            type_to_dup_suffix(&type_to_sub_type(out_t))
        ),
        OutLaneNox => format!(
            "{}{}",
            current_name,
            type_to_lane_suffix(&type_to_sub_type(out_t))
        ),
        In1LaneNox => format!(
            "{}{}",
            current_name,
            type_to_lane_suffix(&type_to_sub_type(in_t[1]))
        ),
        Lane => format!(
            "{}{}",
            current_name,
            type_to_lane_suffixes(out_t, in_t[1], false)
        ),
        In2 => format!("{}{}", current_name, type_to_suffix(in_t[2])),
        In2Lane => format!(
            "{}{}",
            current_name,
            type_to_lane_suffixes(out_t, in_t[2], false)
        ),
        OutLane => format!(
            "{}{}",
            current_name,
            type_to_lane_suffixes(out_t, in_t[2], true)
        ),
        Rot => type_to_rot_suffix(current_name, type_to_suffix(out_t)),
        RotLane => type_to_rot_suffix(current_name, &type_to_lane_suffixes(out_t, in_t[2], false)),
    };
    let current_target = match target {
        Default => "neon",
        ArmV7 => "v7",
        Vfp4 => "vfp4",
        FPArmV8 => "fp-armv8,v8",
        AES => "neon,aes",
        FCMA => "neon,fcma",
        Dotprod => "neon,dotprod",
        I8MM => "neon,i8mm",
        SHA3 => "neon,sha3",
        RDM => "rdm",
        SM4 => "neon,sm4",
        FTTS => "neon,frintts",
    };
    let current_fn = if let Some(current_fn) = current_fn.clone() {
        if link_aarch64.is_some() {
            panic!("[{}] Can't specify link and fn at the same time.", name)
        }
        current_fn
    } else if link_aarch64.is_some() {
        format!("{}_", name)
    } else {
        if multi_fn.is_empty() {
            panic!(
                "[{}] Either (multi) fn or link-aarch have to be specified.",
                name
            )
        }
        String::new()
    };
    let current_aarch64 = current_aarch64.clone().unwrap();
    let mut link_t: Vec<String> = vec![
        in_t[0].to_string(),
        in_t[1].to_string(),
        in_t[2].to_string(),
        out_t.to_string(),
    ];
    let mut ext_c = String::new();
    if let Some(mut link_aarch64) = link_aarch64.clone() {
        if link_aarch64.contains(":") {
            let links: Vec<_> = link_aarch64.split(':').map(|v| v.to_string()).collect();
            assert_eq!(links.len(), 5);
            link_aarch64 = links[0].to_string();
            link_t = vec![
                links[1].clone(),
                links[2].clone(),
                links[3].clone(),
                links[4].clone(),
            ];
        }
        let link_aarch64 = if link_aarch64.starts_with("llvm") {
            ext(&link_aarch64, in_t, out_t)
        } else {
            let mut link = String::from("llvm.aarch64.neon.");
            link.push_str(&link_aarch64);
            ext(&link, in_t, out_t)
        };
        let (ext_inputs, ext_output) = {
            if const_aarch64.is_some() {
                if !matches!(fn_type, Fntype::Normal) {
                    let ptr_type = match fn_type {
                        Fntype::Load => "*const i8",
                        Fntype::Store => "*mut i8",
                        _ => panic!("unsupported fn type"),
                    };
                    let sub = type_to_sub_type(in_t[1]);
                    (
                        match type_sub_len(in_t[1]) {
                            1 => format!("a: {}, n: i64, ptr: {}", sub, ptr_type),
                            2 => format!("a: {}, b: {}, n: i64, ptr: {}", sub, sub, ptr_type),
                            3 => format!(
                                "a: {}, b: {}, c: {}, n: i64, ptr: {}",
                                sub, sub, sub, ptr_type
                            ),
                            4 => format!(
                                "a: {}, b: {}, c: {}, d: {}, n: i64, ptr: {}",
                                sub, sub, sub, sub, ptr_type
                            ),
                            _ => panic!("unsupported type: {}", in_t[1]),
                        },
                        if out_t != "void" {
                            format!(" -> {}", out_t)
                        } else {
                            String::new()
                        },
                    )
                } else {
                    (
                        match para_num {
                            1 => format!("a: {}, n: i32", in_t[0]),
                            2 => format!("a: {}, b: {}, n: i32", in_t[0], in_t[1]),
                            3 => format!("a: {}, b: {}, c: {}, n: i32", in_t[0], in_t[1], in_t[2]),
                            _ => unimplemented!("unknown para_num"),
                        },
                        format!(" -> {}", out_t),
                    )
                }
            } else if matches!(fn_type, Fntype::Store) {
                let sub = type_to_sub_type(in_t[1]);
                let ptr_type = if is_vstx(&name) {
                    "i8".to_string()
                } else {
                    type_to_native_type(in_t[1])
                };
                let subs = match type_sub_len(in_t[1]) {
                    1 => format!("a: {}", sub),
                    2 => format!("a: {}, b: {}", sub, sub),
                    3 => format!("a: {}, b: {}, c: {}", sub, sub, sub),
                    4 => format!("a: {}, b: {}, c: {}, d: {}", sub, sub, sub, sub),
                    _ => panic!("unsupported type: {}", in_t[1]),
                };
                (format!("{}, ptr: *mut {}", subs, ptr_type), String::new())
            } else if is_vldx(&name) {
                let ptr_type = if name.contains("dup") {
                    type_to_native_type(out_t)
                } else {
                    type_to_sub_type(out_t)
                };
                (
                    format!("ptr: *const {}", ptr_type),
                    format!(" -> {}", out_t),
                )
            } else {
                (
                    match para_num {
                        1 => format!("a: {}", link_t[0]),
                        2 => format!("a: {}, b: {}", link_t[0], link_t[1]),
                        3 => format!("a: {}, b: {}, c: {}", link_t[0], link_t[1], link_t[2]),
                        _ => unimplemented!("unknown para_num"),
                    },
                    format!(" -> {}", link_t[3]),
                )
            }
        };
        ext_c = format!(
            r#"#[allow(improper_ctypes)]
    extern "unadjusted" {{
        #[cfg_attr(target_arch = "aarch64", link_name = "{}")]
        fn {}({}){};
    }}
    "#,
            link_aarch64, current_fn, ext_inputs, ext_output,
        );
    };
    let const_declare = if let Some(constn) = constn {
        if constn.contains(":") {
            let constns: Vec<_> = constn.split(':').map(|v| v.to_string()).collect();
            assert_eq!(constns.len(), 2);
            format!(r#"<const {}: i32, const {}: i32>"#, constns[0], constns[1])
        } else {
            format!(r#"<const {}: i32>"#, constn)
        }
    } else {
        String::new()
    };
    let multi_calls = if !multi_fn.is_empty() {
        let mut calls = String::new();
        for i in 0..multi_fn.len() {
            if i > 0 {
                calls.push_str("\n    ");
            }
            calls.push_str(&get_call(
                &multi_fn[i],
                current_name,
                &const_declare,
                in_t,
                out_t,
                fixed,
                None,
                true,
            ));
        }
        calls
    } else {
        String::new()
    };
    let const_assert = if let Some(constn) = constn {
        if constn.contains(":") {
            let constns: Vec<_> = constn.split(':').map(|v| v.to_string()).collect();
            let const_test = current_tests[0].3.as_ref().unwrap();
            let const_tests: Vec<_> = const_test.split(':').map(|v| v.to_string()).collect();
            assert_eq!(constns.len(), 2);
            assert_eq!(const_tests.len(), 2);
            format!(
                r#", {} = {}, {} = {}"#,
                constns[0],
                map_val(in_t[1], &const_tests[0]),
                constns[1],
                map_val(in_t[1], &const_tests[1]),
            )
        } else {
            format!(
                r#", {} = {}"#,
                constn,
                map_val(in_t[1], current_tests[0].3.as_ref().unwrap())
            )
        }
    } else {
        String::new()
    };
    let const_legacy = if let Some(constn) = constn {
        if constn.contains(":") {
            format!(
                "\n#[rustc_legacy_const_generics({}, {})]",
                para_num - 1,
                para_num + 1
            )
        } else {
            format!("\n#[rustc_legacy_const_generics({})]", para_num)
        }
    } else {
        String::new()
    };
    let fn_decl = {
        let fn_output = if out_t == "void" {
            String::new()
        } else {
            format!("-> {} ", out_t)
        };
        let fn_inputs = match para_num {
            1 => format!("(a: {})", in_t[0]),
            2 => format!("(a: {}, b: {})", in_t[0], in_t[1]),
            3 => format!("(a: {}, b: {}, c: {})", in_t[0], in_t[1], in_t[2]),
            _ => panic!("unsupported parameter number"),
        };
        format!(
            "pub unsafe fn {}{}{} {}",
            name, const_declare, fn_inputs, fn_output
        )
    };
    let call_params = {
        if let (Some(const_aarch64), Some(_)) = (const_aarch64, link_aarch64) {
            if !matches!(fn_type, Fntype::Normal) {
                let subs = match type_sub_len(in_t[1]) {
                    1 => "b",
                    2 => "b.0, b.1",
                    3 => "b.0, b.1, b.2",
                    4 => "b.0, b.1, b.2, b.3",
                    _ => panic!("unsupported type: {}", in_t[1]),
                };
                format!(
                    r#"{}
    {}{}({}, {} as i64, a.cast())"#,
                    multi_calls,
                    ext_c,
                    current_fn,
                    subs,
                    constn.as_deref().unwrap()
                )
            } else {
                match para_num {
                    1 => format!(
                        r#"{}
    {}{}(a, {})"#,
                        multi_calls, ext_c, current_fn, const_aarch64
                    ),
                    2 => format!(
                        r#"{}
    {}{}(a, b, {})"#,
                        multi_calls, ext_c, current_fn, const_aarch64
                    ),
                    _ => String::new(),
                }
            }
        } else if link_aarch64.is_some() && matches!(fn_type, Fntype::Store) {
            let cast = if is_vstx(&name) { ".cast()" } else { "" };
            match type_sub_len(in_t[1]) {
                1 => format!(r#"{}{}(b, a{})"#, ext_c, current_fn, cast),
                2 => format!(r#"{}{}(b.0, b.1, a{})"#, ext_c, current_fn, cast),
                3 => format!(r#"{}{}(b.0, b.1, b.2, a{})"#, ext_c, current_fn, cast),
                4 => format!(r#"{}{}(b.0, b.1, b.2, b.3, a{})"#, ext_c, current_fn, cast),
                _ => panic!("unsupported type: {}", in_t[1]),
            }
        } else if link_aarch64.is_some() && is_vldx(&name) {
            format!(r#"{}{}(a.cast())"#, ext_c, current_fn,)
        } else {
            let trans: [&str; 2] = if link_t[3] != out_t {
                ["transmute(", ")"]
            } else {
                ["", ""]
            };
            match (multi_calls.len(), para_num, fixed.len()) {
                (0, 1, 0) => format!(r#"{}{}{}(a){}"#, ext_c, trans[0], current_fn, trans[1]),
                (0, 1, _) => {
                    let fixed: Vec<String> =
                        fixed.iter().take(type_len(in_t[0])).cloned().collect();
                    format!(
                        r#"let b{};
    {}{}{}(a, transmute(b)){}"#,
                        values(in_t[0], &fixed),
                        ext_c,
                        trans[0],
                        current_fn,
                        trans[1],
                    )
                }
                (0, 2, _) => format!(r#"{}{}{}(a, b){}"#, ext_c, trans[0], current_fn, trans[1],),
                (0, 3, _) => format!(r#"{}{}(a, b, c)"#, ext_c, current_fn,),
                (_, 1, _) => format!(r#"{}{}"#, ext_c, multi_calls,),
                (_, 2, _) => format!(r#"{}{}"#, ext_c, multi_calls,),
                (_, 3, _) => format!(r#"{}{}"#, ext_c, multi_calls,),
                (_, _, _) => String::new(),
            }
        }
    };
    let function = format!(
        r#"
{}
#[inline]
#[target_feature(enable = "{}")]
#[cfg_attr(test, assert_instr({}{}))]{}
{}{{
    {}
}}
"#,
        current_comment,
        current_target,
        current_aarch64,
        const_assert,
        const_legacy,
        fn_decl,
        call_params
    );
    let test_target = match target {
        I8MM => "neon,i8mm",
        SM4 => "neon,sm4",
        SHA3 => "neon,sha3",
        FTTS => "neon,frintts",
        _ => "neon",
    };
    let test = match fn_type {
        Fntype::Normal => gen_test(
            &name,
            in_t,
            &out_t,
            current_tests,
            [type_len(in_t[0]), type_len(in_t[1]), type_len(in_t[2])],
            type_len(out_t),
            para_num,
            test_target,
        ),
        Fntype::Load => gen_load_test(&name, in_t, &out_t, current_tests, type_len(out_t)),
        Fntype::Store => gen_store_test(&name, in_t, &out_t, current_tests, type_len(in_t[1])),
    };
    (function, test)
}

fn gen_load_test(
    name: &str,
    in_t: &[&str; 3],
    out_t: &str,
    current_tests: &[(
        Vec<String>,
        Vec<String>,
        Vec<String>,
        Option<String>,
        Vec<String>,
    )],
    type_len: usize,
) -> String {
    let mut test = format!(
        r#"
    #[simd_test(enable = "neon")]
    unsafe fn test_{}() {{"#,
        name,
    );
    for (a, b, _, n, e) in current_tests {
        let a: Vec<String> = a.iter().take(type_len + 1).cloned().collect();
        let e: Vec<String> = e.iter().take(type_len).cloned().collect();
        let has_b = b.len() > 0;
        let has_n = n.is_some();
        let mut input = String::from("[");
        for i in 0..type_len + 1 {
            if i != 0 {
                input.push_str(", ");
            }
            input.push_str(&a[i])
        }
        input.push_str("]");
        let output = |v: &Vec<String>| {
            let mut output = String::from("[");
            for i in 0..type_sub_len(out_t) {
                if i != 0 {
                    output.push_str(", ");
                }
                let sub_len = type_len / type_sub_len(out_t);
                if type_to_global_type(out_t) != "f64" {
                    let mut sub_output = format!("{}::new(", type_to_global_type(out_t));
                    for j in 0..sub_len {
                        if j != 0 {
                            sub_output.push_str(", ");
                        }
                        sub_output.push_str(&v[i * sub_len + j]);
                    }
                    sub_output.push_str(")");
                    output.push_str(&sub_output);
                } else {
                    output.push_str(&v[i]);
                }
            }
            output.push_str("]");
            output
        };
        let input_b = if has_b {
            let b: Vec<String> = b.iter().take(type_len).cloned().collect();
            format!(
                r#"
        let b: [{}; {}] = {};"#,
                type_to_global_type(in_t[1]),
                type_sub_len(in_t[1]),
                output(&b),
            )
        } else {
            String::new()
        };
        let t = format!(
            r#"
        let a: [{}; {}] = {};{}
        let e: [{}; {}] = {};
        let r: [{}; {}] = transmute({}{}(a[1..].as_ptr(){}));
        assert_eq!(r, e);
"#,
            type_to_native_type(out_t),
            type_len + 1,
            input,
            input_b,
            type_to_global_type(out_t),
            type_sub_len(out_t),
            output(&e),
            type_to_global_type(out_t),
            type_sub_len(out_t),
            name,
            if has_n {
                format!("::<{}>", n.as_deref().unwrap())
            } else {
                String::new()
            },
            if has_b { ", transmute(b)" } else { "" },
        );
        test.push_str(&t);
    }
    test.push_str("    }\n");
    test
}

fn gen_store_test(
    name: &str,
    in_t: &[&str; 3],
    _out_t: &str,
    current_tests: &[(
        Vec<String>,
        Vec<String>,
        Vec<String>,
        Option<String>,
        Vec<String>,
    )],
    type_len: usize,
) -> String {
    let mut test = format!(
        r#"
    #[simd_test(enable = "neon")]
    unsafe fn test_{}() {{"#,
        name,
    );
    for (a, _, _, constn, e) in current_tests {
        let a: Vec<String> = a.iter().take(type_len + 1).cloned().collect();
        let e: Vec<String> = e.iter().take(type_len).cloned().collect();
        let mut input = String::from("[");
        for i in 0..type_len + 1 {
            if i != 0 {
                input.push_str(", ");
            }
            input.push_str(&a[i])
        }
        input.push_str("]");
        let mut output = String::from("[");
        for i in 0..type_len {
            if i != 0 {
                output.push_str(", ");
            }
            output.push_str(&e[i])
        }
        output.push_str("]");
        let const_n = constn
            .as_deref()
            .map_or(String::new(), |n| format!("::<{}>", n.to_string()));
        let t = format!(
            r#"
        let a: [{}; {}] = {};
        let e: [{}; {}] = {};
        let mut r: [{}; {}] = [0{}; {}];
        {}{}(r.as_mut_ptr(), core::ptr::read_unaligned(a[1..].as_ptr().cast()));
        assert_eq!(r, e);
"#,
            type_to_native_type(in_t[1]),
            type_len + 1,
            input,
            type_to_native_type(in_t[1]),
            type_len,
            output,
            type_to_native_type(in_t[1]),
            type_len,
            type_to_native_type(in_t[1]),
            type_len,
            name,
            const_n,
        );
        test.push_str(&t);
    }
    test.push_str("    }\n");
    test
}

fn gen_test(
    name: &str,
    in_t: &[&str; 3],
    out_t: &str,
    current_tests: &[(
        Vec<String>,
        Vec<String>,
        Vec<String>,
        Option<String>,
        Vec<String>,
    )],
    len_in: [usize; 3],
    len_out: usize,
    para_num: i32,
    target: &str,
) -> String {
    let mut test = format!(
        r#"
    #[simd_test(enable = "{}")]
    unsafe fn test_{}() {{"#,
        target, name,
    );
    for (a, b, c, n, e) in current_tests {
        let a: Vec<String> = a.iter().take(len_in[0]).cloned().collect();
        let b: Vec<String> = b.iter().take(len_in[1]).cloned().collect();
        let c: Vec<String> = c.iter().take(len_in[2]).cloned().collect();
        let e: Vec<String> = e.iter().take(len_out).cloned().collect();
        let const_value = if let Some(constn) = n {
            if constn.contains(":") {
                let constns: Vec<_> = constn.split(':').map(|v| v.to_string()).collect();
                format!(
                    r#"::<{}, {}>"#,
                    map_val(in_t[1], &constns[0]),
                    map_val(in_t[1], &constns[1])
                )
            } else {
                format!(r#"::<{}>"#, map_val(in_t[1], constn))
            }
        } else {
            String::new()
        };
        let r_type = match type_sub_len(out_t) {
            1 => type_to_global_type(out_t).to_string(),
            _ => format!("[{}; {}]", type_to_native_type(out_t), type_len(out_t)),
        };
        let t = {
            match para_num {
                1 => {
                    format!(
                        r#"
        let a{};
        let e{};
        let r: {} = transmute({}{}(transmute(a)));
        assert_eq!(r, e);
"#,
                        values(in_t[0], &a),
                        values(out_t, &e),
                        r_type,
                        name,
                        const_value
                    )
                }
                2 => {
                    format!(
                        r#"
        let a{};
        let b{};
        let e{};
        let r: {} = transmute({}{}(transmute(a), transmute(b)));
        assert_eq!(r, e);
"#,
                        values(in_t[0], &a),
                        values(in_t[1], &b),
                        values(out_t, &e),
                        r_type,
                        name,
                        const_value
                    )
                }
                3 => {
                    format!(
                        r#"
        let a{};
        let b{};
        let c{};
        let e{};
        let r: {} = transmute({}{}(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
"#,
                        values(in_t[0], &a),
                        values(in_t[1], &b),
                        values(in_t[2], &c),
                        values(out_t, &e),
                        r_type,
                        name,
                        const_value
                    )
                }
                _ => {
                    panic!("no support para_num:{}", para_num.to_string())
                }
            }
        };

        test.push_str(&t);
    }
    test.push_str("    }\n");
    test
}

#[allow(clippy::too_many_arguments)]
fn gen_arm(
    current_comment: &str,
    current_fn: &Option<String>,
    current_name: &str,
    current_arm: &str,
    link_arm: &Option<String>,
    current_aarch64: &Option<String>,
    link_aarch64: &Option<String>,
    const_arm: &Option<String>,
    const_aarch64: &Option<String>,
    constn: &Option<String>,
    in_t: &[&str; 3],
    out_t: &str,
    current_tests: &[(
        Vec<String>,
        Vec<String>,
        Vec<String>,
        Option<String>,
        Vec<String>,
    )],
    suffix: Suffix,
    para_num: i32,
    target: TargetFeature,
    fixed: &Vec<String>,
    multi_fn: &Vec<String>,
    fn_type: Fntype,
    separate: bool,
) -> (String, String) {
    let name = match suffix {
        Normal => format!("{}{}", current_name, type_to_suffix(in_t[1])),
        NoQ => format!("{}{}", current_name, type_to_noq_suffix(in_t[1])),
        Double => format!(
            "{}{}",
            current_name,
            type_to_double_suffixes(out_t, in_t[1])
        ),
        NoQDouble => format!(
            "{}{}",
            current_name,
            type_to_noq_double_suffixes(out_t, in_t[1])
        ),
        NSuffix => format!("{}{}", current_name, type_to_n_suffix(in_t[1])),
        DoubleN => format!(
            "{}{}",
            current_name,
            type_to_double_n_suffixes(out_t, in_t[1])
        ),
        NoQNSuffix => format!("{}{}", current_name, type_to_noq_n_suffix(in_t[1])),
        OutSuffix => format!("{}{}", current_name, type_to_suffix(out_t)),
        OutNSuffix => format!("{}{}", current_name, type_to_n_suffix(out_t)),
        OutNox => format!(
            "{}{}",
            current_name,
            type_to_suffix(&type_to_sub_type(out_t))
        ),
        In1Nox => format!(
            "{}{}",
            current_name,
            type_to_suffix(&type_to_sub_type(in_t[1]))
        ),
        OutDupNox => format!(
            "{}{}",
            current_name,
            type_to_dup_suffix(&type_to_sub_type(out_t))
        ),
        OutLaneNox => format!(
            "{}{}",
            current_name,
            type_to_lane_suffix(&type_to_sub_type(out_t))
        ),
        In1LaneNox => format!(
            "{}{}",
            current_name,
            type_to_lane_suffix(&type_to_sub_type(in_t[1]))
        ),
        Lane => format!(
            "{}{}",
            current_name,
            type_to_lane_suffixes(out_t, in_t[1], false)
        ),
        In2 => format!("{}{}", current_name, type_to_suffix(in_t[2])),
        In2Lane => format!(
            "{}{}",
            current_name,
            type_to_lane_suffixes(out_t, in_t[2], false)
        ),
        OutLane => format!(
            "{}{}",
            current_name,
            type_to_lane_suffixes(out_t, in_t[2], true)
        ),
        Rot => type_to_rot_suffix(current_name, type_to_suffix(out_t)),
        RotLane => type_to_rot_suffix(current_name, &type_to_lane_suffixes(out_t, in_t[2], false)),
    };
    let current_aarch64 = current_aarch64
        .clone()
        .unwrap_or_else(|| current_arm.to_string());
    let current_target_aarch64 = match target {
        Default => "neon",
        ArmV7 => "neon",
        Vfp4 => "neon",
        FPArmV8 => "neon",
        AES => "neon,aes",
        FCMA => "neon,fcma",
        Dotprod => "neon,dotprod",
        I8MM => "neon,i8mm",
        SHA3 => "neon,sha3",
        RDM => "rdm",
        SM4 => "neon,sm4",
        FTTS => "neon,frintts",
    };
    let current_target_arm = match target {
        Default => "v7",
        ArmV7 => "v7",
        Vfp4 => "vfp4",
        FPArmV8 => "fp-armv8,v8",
        AES => "aes,v8",
        FCMA => "v8",    // v8.3a
        Dotprod => "v8", // v8.2a
        I8MM => "v8,i8mm",
        RDM => unreachable!(),
        SM4 => unreachable!(),
        SHA3 => unreachable!(),
        FTTS => unreachable!(),
    };
    let current_fn = if let Some(current_fn) = current_fn.clone() {
        if link_aarch64.is_some() || link_arm.is_some() {
            panic!(
                "[{}] Can't specify link and function at the same time. {} / {:?} / {:?}",
                name, current_fn, link_aarch64, link_arm
            )
        }
        current_fn
    } else if link_aarch64.is_some() || link_arm.is_some() {
        format!("{}_", name)
    } else {
        if multi_fn.is_empty() {
            panic!(
                "[{}] Either fn or link-arm and link-aarch have to be specified.",
                name
            )
        }
        String::new()
    };
    let mut ext_c = String::new();
    let mut ext_c_arm = if multi_fn.is_empty() || link_arm.is_none() {
        String::new()
    } else {
        String::from(
            r#"
    "#,
        )
    };
    let mut ext_c_aarch64 = if multi_fn.is_empty() || link_aarch64.is_none() {
        String::new()
    } else {
        String::from(
            r#"
    "#,
        )
    };
    let mut link_arm_t: Vec<String> = vec![
        in_t[0].to_string(),
        in_t[1].to_string(),
        in_t[2].to_string(),
        out_t.to_string(),
    ];
    let mut link_aarch64_t: Vec<String> = vec![
        in_t[0].to_string(),
        in_t[1].to_string(),
        in_t[2].to_string(),
        out_t.to_string(),
    ];
    if let (Some(mut link_arm), Some(mut link_aarch64)) = (link_arm.clone(), link_aarch64.clone()) {
        if link_arm.contains(":") {
            let links: Vec<_> = link_arm.split(':').map(|v| v.to_string()).collect();
            assert_eq!(links.len(), 5);
            link_arm = links[0].to_string();
            link_arm_t = vec![
                links[1].clone(),
                links[2].clone(),
                links[3].clone(),
                links[4].clone(),
            ];
        }
        if link_aarch64.contains(":") {
            let links: Vec<_> = link_aarch64.split(':').map(|v| v.to_string()).collect();
            assert_eq!(links.len(), 5);
            link_aarch64 = links[0].to_string();
            link_aarch64_t = vec![
                links[1].clone(),
                links[2].clone(),
                links[3].clone(),
                links[4].clone(),
            ];
        }
        let link_arm = if link_arm.starts_with("llvm") {
            ext(&link_arm, in_t, out_t)
        } else {
            let mut link = String::from("llvm.arm.neon.");
            link.push_str(&link_arm);
            ext(&link, in_t, out_t)
        };
        let link_aarch64 = if link_aarch64.starts_with("llvm") {
            ext(&link_aarch64, in_t, out_t)
        } else {
            let mut link = String::from("llvm.aarch64.neon.");
            link.push_str(&link_aarch64);
            ext(&link, in_t, out_t)
        };
        if out_t == link_arm_t[3] && out_t == link_aarch64_t[3] {
            ext_c = format!(
                r#"#[allow(improper_ctypes)]
    extern "unadjusted" {{
        #[cfg_attr(target_arch = "arm", link_name = "{}")]
        #[cfg_attr(target_arch = "aarch64", link_name = "{}")]
        fn {}({}) -> {};
    }}
"#,
                link_arm,
                link_aarch64,
                current_fn,
                match para_num {
                    1 => format!("a: {}", in_t[0]),
                    2 => format!("a: {}, b: {}", in_t[0], in_t[1]),
                    3 => format!("a: {}, b: {}, c: {}", in_t[0], in_t[1], in_t[2]),
                    _ => unimplemented!("unknown para_num"),
                },
                out_t
            );
        };
        let (arm_ext_inputs, arm_ext_output) = {
            if let Some(const_arm) = const_arm {
                if !matches!(fn_type, Fntype::Normal) {
                    let ptr_type = match fn_type {
                        Fntype::Load => "*const i8",
                        Fntype::Store => "*mut i8",
                        _ => panic!("unsupported fn type"),
                    };
                    let sub_type = type_to_sub_type(in_t[1]);
                    let inputs = match type_sub_len(in_t[1]) {
                        1 => format!("a: {}", sub_type),
                        2 => format!("a: {}, b: {}", sub_type, sub_type,),
                        3 => format!("a: {}, b: {}, c: {}", sub_type, sub_type, sub_type,),
                        4 => format!(
                            "a: {}, b: {}, c: {}, d: {}",
                            sub_type, sub_type, sub_type, sub_type,
                        ),
                        _ => panic!("unknown type: {}", in_t[1]),
                    };
                    let out = if out_t == "void" {
                        String::new()
                    } else {
                        format!(" -> {}", out_t)
                    };
                    (
                        format!("ptr: {}, {}, n: i32, size: i32", ptr_type, inputs),
                        out,
                    )
                } else {
                    let (_, const_type) = if const_arm.contains(":") {
                        let consts: Vec<_> =
                            const_arm.split(':').map(|v| v.trim().to_string()).collect();
                        (consts[0].clone(), consts[1].clone())
                    } else {
                        (
                            const_arm.to_string(),
                            in_t[para_num as usize - 1].to_string(),
                        )
                    };
                    (
                        match para_num {
                            1 => format!("a: {}, n: {}", in_t[0], const_type),
                            2 => format!("a: {}, b: {}, n: {}", in_t[0], in_t[1], const_type),
                            3 => format!(
                                "a: {}, b: {}, c: {}, n: {}",
                                in_t[0], in_t[1], in_t[2], const_type
                            ),
                            _ => unimplemented!("unknown para_num"),
                        },
                        format!(" -> {}", out_t),
                    )
                }
            } else if out_t != link_arm_t[3] {
                (
                    match para_num {
                        1 => format!("a: {}", link_arm_t[0]),
                        2 => format!("a: {}, b: {}", link_arm_t[0], link_arm_t[1]),
                        3 => format!(
                            "a: {}, b: {}, c: {}",
                            link_arm_t[0], link_arm_t[1], link_arm_t[2]
                        ),
                        _ => unimplemented!("unknown para_num"),
                    },
                    format!(" -> {}", link_arm_t[3]),
                )
            } else if matches!(fn_type, Fntype::Store) {
                let sub_type = type_to_sub_type(in_t[1]);
                let inputs = match type_sub_len(in_t[1]) {
                    1 => format!("a: {}", sub_type),
                    2 => format!("a: {}, b: {}", sub_type, sub_type,),
                    3 => format!("a: {}, b: {}, c: {}", sub_type, sub_type, sub_type,),
                    4 => format!(
                        "a: {}, b: {}, c: {}, d: {}",
                        sub_type, sub_type, sub_type, sub_type,
                    ),
                    _ => panic!("unknown type: {}", in_t[1]),
                };
                let (ptr_type, size) = if is_vstx(&name) {
                    ("i8".to_string(), ", size: i32")
                } else {
                    (type_to_native_type(in_t[1]), "")
                };
                (
                    format!("ptr: *mut {}, {}{}", ptr_type, inputs, size),
                    String::new(),
                )
            } else if is_vldx(&name) {
                (
                    format!("ptr: *const i8, size: i32"),
                    format!(" -> {}", out_t),
                )
            } else {
                (String::new(), String::new())
            }
        };
        ext_c_arm.push_str(&format!(
            r#"#[allow(improper_ctypes)]
    extern "unadjusted" {{
        #[cfg_attr(target_arch = "arm", link_name = "{}")]
        fn {}({}){};
    }}
"#,
            link_arm, current_fn, arm_ext_inputs, arm_ext_output,
        ));
        let (aarch64_ext_inputs, aarch64_ext_output) = {
            if const_aarch64.is_some() {
                if !matches!(fn_type, Fntype::Normal) {
                    let ptr_type = match fn_type {
                        Fntype::Load => "*const i8",
                        Fntype::Store => "*mut i8",
                        _ => panic!("unsupported fn type"),
                    };
                    let sub_type = type_to_sub_type(in_t[1]);
                    let mut inputs = match type_sub_len(in_t[1]) {
                        1 => format!("a: {}", sub_type,),
                        2 => format!("a: {}, b: {}", sub_type, sub_type,),
                        3 => format!("a: {}, b: {}, c: {}", sub_type, sub_type, sub_type,),
                        4 => format!(
                            "a: {}, b: {}, c: {}, d: {}",
                            sub_type, sub_type, sub_type, sub_type,
                        ),
                        _ => panic!("unknown type: {}", in_t[1]),
                    };
                    inputs.push_str(&format!(", n: i64, ptr: {}", ptr_type));
                    let out = if out_t == "void" {
                        String::new()
                    } else {
                        format!(" -> {}", out_t)
                    };
                    (inputs, out)
                } else {
                    (
                        match para_num {
                            1 => format!("a: {}, n: i32", in_t[0]),
                            2 => format!("a: {}, b: {}, n: i32", in_t[0], in_t[1]),
                            3 => format!("a: {}, b: {}, c: {}, n: i32", in_t[0], in_t[1], in_t[2]),
                            _ => unimplemented!("unknown para_num"),
                        },
                        format!(" -> {}", out_t),
                    )
                }
            } else if out_t != link_aarch64_t[3] {
                (
                    match para_num {
                        1 => format!("a: {}", link_aarch64_t[0]),
                        2 => format!("a: {}, b: {}", link_aarch64_t[0], link_aarch64_t[1]),
                        3 => format!(
                            "a: {}, b: {}, c: {}",
                            link_aarch64_t[0], link_aarch64_t[1], link_aarch64_t[2]
                        ),
                        _ => unimplemented!("unknown para_num"),
                    },
                    format!(" -> {}", link_aarch64_t[3]),
                )
            } else if matches!(fn_type, Fntype::Store) {
                let sub_type = type_to_sub_type(in_t[1]);
                let mut inputs = match type_sub_len(in_t[1]) {
                    1 => format!("a: {}", sub_type,),
                    2 => format!("a: {}, b: {}", sub_type, sub_type,),
                    3 => format!("a: {}, b: {}, c: {}", sub_type, sub_type, sub_type,),
                    4 => format!(
                        "a: {}, b: {}, c: {}, d: {}",
                        sub_type, sub_type, sub_type, sub_type,
                    ),
                    _ => panic!("unknown type: {}", in_t[1]),
                };
                let ptr_type = if is_vstx(&name) {
                    "i8".to_string()
                } else {
                    type_to_native_type(in_t[1])
                };
                inputs.push_str(&format!(", ptr: *mut {}", ptr_type));
                (inputs, String::new())
            } else if is_vldx(&name) {
                let ptr_type = if name.contains("dup") {
                    type_to_native_type(out_t)
                } else {
                    type_to_sub_type(out_t)
                };
                (
                    format!("ptr: *const {}", ptr_type),
                    format!(" -> {}", out_t),
                )
            } else {
                (String::new(), String::new())
            }
        };
        ext_c_aarch64.push_str(&format!(
            r#"#[allow(improper_ctypes)]
    extern "unadjusted" {{
        #[cfg_attr(target_arch = "aarch64", link_name = "{}")]
        fn {}({}){};
    }}
"#,
            link_aarch64, current_fn, aarch64_ext_inputs, aarch64_ext_output,
        ));
    };
    let const_declare = if let Some(constn) = constn {
        format!(r#"<const {}: i32>"#, constn)
    } else {
        String::new()
    };
    let multi_calls = if !multi_fn.is_empty() {
        let mut calls = String::new();
        for i in 0..multi_fn.len() {
            if i > 0 {
                calls.push_str("\n    ");
            }
            calls.push_str(&get_call(
                &multi_fn[i],
                current_name,
                &const_declare,
                in_t,
                out_t,
                fixed,
                None,
                false,
            ));
        }
        calls
    } else {
        String::new()
    };
    let const_assert = if let Some(constn) = constn {
        format!(
            r#", {} = {}"#,
            constn,
            map_val(in_t[1], current_tests[0].3.as_ref().unwrap())
        )
    } else {
        String::new()
    };
    let const_legacy = if constn.is_some() {
        format!("\n#[rustc_legacy_const_generics({})]", para_num)
    } else {
        String::new()
    };
    let fn_decl = {
        let fn_output = if out_t == "void" {
            String::new()
        } else {
            format!("-> {} ", out_t)
        };
        let fn_inputs = match para_num {
            1 => format!("(a: {})", in_t[0]),
            2 => format!("(a: {}, b: {})", in_t[0], in_t[1]),
            3 => format!("(a: {}, b: {}, c: {})", in_t[0], in_t[1], in_t[2]),
            _ => panic!("unsupported parameter number"),
        };
        format!(
            "pub unsafe fn {}{}{} {}",
            name, const_declare, fn_inputs, fn_output
        )
    };
    let function = if separate {
        let call_arm = {
            let arm_params = if let (Some(const_arm), Some(_)) = (const_arm, link_arm) {
                if !matches!(fn_type, Fntype::Normal) {
                    let subs = match type_sub_len(in_t[1]) {
                        1 => "b",
                        2 => "b.0, b.1",
                        3 => "b.0, b.1, b.2",
                        4 => "b.0, b.1, b.2, b.3",
                        _ => "",
                    };
                    format!(
                        "{}(a.cast(), {}, {}, {})",
                        current_fn,
                        subs,
                        constn.as_deref().unwrap(),
                        type_bits(&type_to_sub_type(in_t[1])) / 8,
                    )
                } else {
                    let cnt = if const_arm.contains(':') {
                        let consts: Vec<_> =
                            const_arm.split(':').map(|v| v.trim().to_string()).collect();
                        consts[0].clone()
                    } else {
                        let const_arm = const_arm.replace("ttn", &type_to_native_type(in_t[1]));
                        let mut cnt = String::from(in_t[1]);
                        cnt.push_str("(");
                        for i in 0..type_len(in_t[1]) {
                            if i != 0 {
                                cnt.push_str(", ");
                            }
                            cnt.push_str(&const_arm);
                        }
                        cnt.push_str(")");
                        cnt
                    };
                    match para_num {
                        1 => format!("{}(a, {})", current_fn, cnt),
                        2 => format!("{}(a, b, {})", current_fn, cnt),
                        _ => String::new(),
                    }
                }
            } else if out_t != link_arm_t[3] {
                match para_num {
                    1 => format!("transmute({}(a))", current_fn,),
                    2 => format!("transmute({}(transmute(a), transmute(b)))", current_fn,),
                    _ => String::new(),
                }
            } else if matches!(fn_type, Fntype::Store) {
                let (cast, size) = if is_vstx(&name) {
                    (
                        ".cast()",
                        format!(", {}", type_bits(&type_to_sub_type(in_t[1])) / 8),
                    )
                } else {
                    ("", String::new())
                };
                match type_sub_len(in_t[1]) {
                    1 => format!("{}(a{}, b{})", current_fn, cast, size),
                    2 => format!("{}(a{}, b.0, b.1{})", current_fn, cast, size),
                    3 => format!("{}(a{}, b.0, b.1, b.2{})", current_fn, cast, size),
                    4 => format!("{}(a{}, b.0, b.1, b.2, b.3{})", current_fn, cast, size),
                    _ => String::new(),
                }
            } else if link_arm.is_some() && is_vldx(&name) {
                format!(
                    "{}(a as *const i8, {})",
                    current_fn,
                    type_bits(&type_to_sub_type(out_t)) / 8
                )
            } else {
                String::new()
            };
            format!(
                r#"{}{{
    {}{}{}
}}"#,
                fn_decl, multi_calls, ext_c_arm, arm_params
            )
        };
        let call_aarch64 = {
            let aarch64_params =
                if let (Some(const_aarch64), Some(_)) = (const_aarch64, link_aarch64) {
                    if !matches!(fn_type, Fntype::Normal) {
                        let subs = match type_sub_len(in_t[1]) {
                            1 => "b",
                            2 => "b.0, b.1",
                            3 => "b.0, b.1, b.2",
                            4 => "b.0, b.1, b.2, b.3",
                            _ => "",
                        };
                        format!(
                            "{}({}, {} as i64, a.cast())",
                            current_fn,
                            subs,
                            constn.as_deref().unwrap()
                        )
                    } else {
                        match para_num {
                            1 => format!("{}(a, {})", current_fn, const_aarch64),
                            2 => format!("{}(a, b, {})", current_fn, const_aarch64),
                            _ => String::new(),
                        }
                    }
                } else if out_t != link_aarch64_t[3] {
                    match para_num {
                        1 => format!("transmute({}(a))", current_fn,),
                        2 => format!("transmute({}(a, b))", current_fn,),
                        _ => String::new(),
                    }
                } else if matches!(fn_type, Fntype::Store) {
                    let cast = if is_vstx(&name) { ".cast()" } else { "" };
                    match type_sub_len(in_t[1]) {
                        1 => format!("{}(b, a{})", current_fn, cast),
                        2 => format!("{}(b.0, b.1, a{})", current_fn, cast),
                        3 => format!("{}(b.0, b.1, b.2, a{})", current_fn, cast),
                        4 => format!("{}(b.0, b.1, b.2, b.3, a{})", current_fn, cast),
                        _ => String::new(),
                    }
                } else if link_aarch64.is_some() && is_vldx(&name) {
                    format!("{}(a.cast())", current_fn)
                } else {
                    String::new()
                };
            format!(
                r#"{}{{
    {}{}{}
}}"#,
                fn_decl, multi_calls, ext_c_aarch64, aarch64_params
            )
        };
        format!(
            r#"
{}
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,{}")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr({}{}))]{}
{}

{}
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "{}")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr({}{}))]{}
{}
"#,
            current_comment,
            current_target_arm,
            expand_intrinsic(&current_arm, in_t[1]),
            const_assert,
            const_legacy,
            call_arm,
            current_comment,
            current_target_aarch64,
            expand_intrinsic(&current_aarch64, in_t[1]),
            const_assert,
            const_legacy,
            call_aarch64,
        )
    } else {
        let call = {
            let stmts = match (multi_calls.len(), para_num, fixed.len()) {
                (0, 1, 0) => format!(r#"{}{}(a)"#, ext_c, current_fn,),
                (0, 1, _) => {
                    let fixed: Vec<String> =
                        fixed.iter().take(type_len(in_t[0])).cloned().collect();
                    format!(
                        r#"let b{};
    {}{}(a, transmute(b))"#,
                        values(in_t[0], &fixed),
                        ext_c,
                        current_fn,
                    )
                }
                (0, 2, _) => format!(r#"{}{}(a, b)"#, ext_c, current_fn,),
                (0, 3, _) => format!(r#"{}{}(a, b, c)"#, ext_c, current_fn,),
                (_, 1, _) => format!(r#"{}{}"#, ext_c, multi_calls,),
                (_, 2, _) => format!(r#"{}{}"#, ext_c, multi_calls,),
                (_, 3, _) => format!(r#"{}{}"#, ext_c, multi_calls,),
                (_, _, _) => String::new(),
            };
            if stmts != String::new() {
                format!(
                    r#"{}{{
    {}
}}"#,
                    fn_decl, stmts
                )
            } else {
                String::new()
            }
        };
        format!(
            r#"
{}
#[inline]
#[target_feature(enable = "{}")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "{}"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr({}{}))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr({}{}))]{}
{}
"#,
            current_comment,
            current_target_aarch64,
            current_target_arm,
            expand_intrinsic(&current_arm, in_t[1]),
            const_assert,
            expand_intrinsic(&current_aarch64, in_t[1]),
            const_assert,
            const_legacy,
            call,
        )
    };
    let test_target = match target {
        I8MM => "neon,i8mm",
        SM4 => "neon,sm4",
        SHA3 => "neon,sha3",
        FTTS => "neon,frintts",
        _ => "neon",
    };
    let test = match fn_type {
        Fntype::Normal => gen_test(
            &name,
            in_t,
            &out_t,
            current_tests,
            [type_len(in_t[0]), type_len(in_t[1]), type_len(in_t[2])],
            type_len(out_t),
            para_num,
            test_target,
        ),
        Fntype::Load => gen_load_test(&name, in_t, &out_t, current_tests, type_len(out_t)),
        Fntype::Store => gen_store_test(&name, in_t, &out_t, current_tests, type_len(in_t[1])),
    };
    (function, test)
}

fn expand_intrinsic(intr: &str, t: &str) -> String {
    if intr.ends_with('.') {
        let ext = match t {
            "int8x8_t" => "i8",
            "int8x16_t" => "i8",
            "int16x4_t" => "i16",
            "int16x8_t" => "i16",
            "int32x2_t" => "i32",
            "int32x4_t" => "i32",
            "int64x1_t" => "i64",
            "int64x2_t" => "i64",
            "uint8x8_t" => "i8",
            "uint8x16_t" => "i8",
            "uint16x4_t" => "i16",
            "uint16x8_t" => "i16",
            "uint32x2_t" => "i32",
            "uint32x4_t" => "i32",
            "uint64x1_t" => "i64",
            "uint64x2_t" => "i64",
            "float16x4_t" => "f16",
            "float16x8_t" => "f16",
            "float32x2_t" => "f32",
            "float32x4_t" => "f32",
            "float64x1_t" => "f64",
            "float64x2_t" => "f64",
            "poly8x8_t" => "i8",
            "poly8x16_t" => "i8",
            "poly16x4_t" => "i16",
            "poly16x8_t" => "i16",
            /*
            "poly64x1_t" => "i64x1",
            "poly64x2_t" => "i64x2",
            */
            _ => panic!("unknown type for extension: {}", t),
        };
        format!(r#""{}{}""#, intr, ext)
    } else if intr.ends_with(".s") {
        let ext = match t {
            "int8x8_t" => "s8",
            "int8x16_t" => "s8",
            "int16x4_t" => "s16",
            "int16x8_t" => "s16",
            "int32x2_t" => "s32",
            "int32x4_t" => "s32",
            "int64x1_t" => "s64",
            "int64x2_t" => "s64",
            "uint8x8_t" => "u8",
            "uint8x16_t" => "u8",
            "uint16x4_t" => "u16",
            "uint16x8_t" => "u16",
            "uint32x2_t" => "u32",
            "uint32x4_t" => "u32",
            "uint64x1_t" => "u64",
            "uint64x2_t" => "u64",
            "poly8x8_t" => "p8",
            "poly8x16_t" => "p8",
            "poly16x4_t" => "p16",
            "poly16x8_t" => "p16",
            "float16x4_t" => "f16",
            "float16x8_t" => "f16",
            "float32x2_t" => "f32",
            "float32x4_t" => "f32",
            "float64x1_t" => "f64",
            "float64x2_t" => "f64",
            /*
            "poly64x1_t" => "i64x1",
            "poly64x2_t" => "i64x2",
            */
            _ => panic!("unknown type for extension: {}", t),
        };
        format!(r#""{}{}""#, &intr[..intr.len() - 1], ext)
    } else if intr.ends_with(".l") {
        let ext = match t {
            "int8x8_t" => "8",
            "int8x16_t" => "8",
            "int16x4_t" => "16",
            "int16x8_t" => "16",
            "int32x2_t" => "32",
            "int32x4_t" => "32",
            "int64x1_t" => "64",
            "int64x2_t" => "64",
            "uint8x8_t" => "8",
            "uint8x16_t" => "8",
            "uint16x4_t" => "16",
            "uint16x8_t" => "16",
            "uint32x2_t" => "32",
            "uint32x4_t" => "32",
            "uint64x1_t" => "64",
            "uint64x2_t" => "64",
            "poly8x8_t" => "8",
            "poly8x16_t" => "8",
            "poly16x4_t" => "16",
            "poly16x8_t" => "16",
            "float16x4_t" => "16",
            "float16x8_t" => "16",
            "float32x2_t" => "32",
            "float32x4_t" => "32",
            "float64x1_t" => "64",
            "float64x2_t" => "64",
            "poly64x1_t" => "64",
            "poly64x2_t" => "64",
            _ => panic!("unknown type for extension: {}", t),
        };
        format!(r#""{}{}""#, &intr[..intr.len() - 1], ext)
    } else {
        intr.to_string()
    }
}

fn get_call(
    in_str: &str,
    current_name: &str,
    const_declare: &str,
    in_t: &[&str; 3],
    out_t: &str,
    fixed: &Vec<String>,
    n: Option<i32>,
    aarch64: bool,
) -> String {
    let params: Vec<_> = in_str.split(',').map(|v| v.trim().to_string()).collect();
    assert!(params.len() > 0);
    let mut fn_name = params[0].clone();
    if fn_name == "a" {
        return String::from("a");
    }
    if fn_name == "transpose-1-in_len" {
        return transpose1(type_len(in_t[1])).to_string();
    }
    if fn_name == "transpose-2-in_len" {
        return transpose2(type_len(in_t[1])).to_string();
    }
    if fn_name == "zip-1-in_len" {
        return zip1(type_len(in_t[1])).to_string();
    }
    if fn_name == "zip-2-in_len" {
        return zip2(type_len(in_t[1])).to_string();
    }
    if fn_name == "unzip-1-in_len" {
        return unzip1(type_len(in_t[1])).to_string();
    }
    if fn_name == "unzip-2-in_len" {
        return unzip2(type_len(in_t[1])).to_string();
    }
    if fn_name.starts_with("dup") {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        let len = match &*fn_format[1] {
            "out_len" => type_len(out_t),
            "in_len" => type_len(in_t[1]),
            "in0_len" => type_len(in_t[0]),
            "halflen" => type_len(in_t[1]) / 2,
            _ => 0,
        };
        let mut s = format!("{} [", const_declare);
        for i in 0..len {
            if i != 0 {
                s.push_str(", ");
            }
            s.push_str(&fn_format[2]);
        }
        s.push_str("]");
        return s;
    }
    if fn_name.starts_with("asc") {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        let start = match &*fn_format[1] {
            "0" => 0,
            "n" => n.unwrap(),
            "out_len" => type_len(out_t) as i32,
            "halflen" => (type_len(in_t[1]) / 2) as i32,
            s => s.parse::<i32>().unwrap(),
        };
        let len = match &*fn_format[2] {
            "out_len" => type_len(out_t),
            "in_len" => type_len(in_t[1]),
            "in0_len" => type_len(in_t[0]),
            "halflen" => type_len(in_t[1]) / 2,
            _ => 0,
        };
        return asc(start, len);
    }
    if fn_name.starts_with("base") {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        assert_eq!(fn_format.len(), 3);
        let mut s = format!("<const {}: i32> [", &fn_format[2]);
        let base_len = fn_format[1].parse::<usize>().unwrap();
        for i in 0..type_len(in_t[1]) / base_len {
            for j in 0..base_len {
                if i != 0 || j != 0 {
                    s.push_str(", ");
                }
                s.push_str(&format!("{} * {} as u32", base_len, &fn_format[2]));
                if j != 0 {
                    s.push_str(&format!(" + {}", j));
                }
            }
        }
        s.push_str("]");
        return s;
    }
    if fn_name.starts_with("as") {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        assert_eq!(fn_format.len(), 3);
        let t = match &*fn_format[2] {
            "in_ttn" => type_to_native_type(in_t[1]),
            _ => String::new(),
        };
        return format!("{} as {}", &fn_format[1], t);
    }
    if fn_name.starts_with("ins") {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        let n = n.unwrap();
        let len = match &*fn_format[1] {
            "out_len" => type_len(out_t),
            "in_len" => type_len(in_t[1]),
            "in0_len" => type_len(in_t[0]),
            _ => 0,
        };
        let offset = match &*fn_format[2] {
            "out_len" => type_len(out_t),
            "in_len" => type_len(in_t[1]),
            "in0_len" => type_len(in_t[0]),
            _ => 0,
        };
        let mut s = format!("{} [", const_declare);
        for i in 0..len {
            if i != 0 {
                s.push_str(", ");
            }
            if i == n as usize {
                s.push_str(&format!("{} + {} as u32", offset.to_string(), fn_format[3]));
            } else {
                s.push_str(&i.to_string());
            }
        }
        s.push_str("]");
        return s;
    }
    if fn_name.starts_with("static_assert_imm") {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        let len = match &*fn_format[1] {
            "out_exp_len" => type_exp_len(out_t, 1),
            "out_bits_exp_len" => type_bits_exp_len(out_t),
            "in_exp_len" => type_exp_len(in_t[1], 1),
            "in_bits_exp_len" => type_bits_exp_len(in_t[1]),
            "in0_exp_len" => type_exp_len(in_t[0], 1),
            "in1_exp_len" => type_exp_len(in_t[1], 1),
            "in2_exp_len" => type_exp_len(in_t[2], 1),
            "in2_rot" => type_exp_len(in_t[2], 2),
            "in2_dot" => type_exp_len(in_t[2], 4),
            _ => 0,
        };
        if len == 0 {
            return format!(
                r#"static_assert!({} : i32 where {} == 0);"#,
                fn_format[2], fn_format[2]
            );
        } else {
            return format!(r#"static_assert_imm{}!({});"#, len, fn_format[2]);
        }
    }
    if fn_name.starts_with("static_assert") {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        let lim1 = if fn_format[2] == "bits" {
            type_bits(in_t[1]).to_string()
        } else if fn_format[2] == "halfbits" {
            (type_bits(in_t[1]) / 2).to_string()
        } else {
            fn_format[2].clone()
        };
        let lim2 = if fn_format[3] == "bits" {
            type_bits(in_t[1]).to_string()
        } else if fn_format[3] == "halfbits" {
            (type_bits(in_t[1]) / 2).to_string()
        } else {
            fn_format[3].clone()
        };
        if lim1 == lim2 {
            return format!(
                r#"static_assert!({} : i32 where {} == {});"#,
                fn_format[1], fn_format[1], lim1
            );
        } else {
            return format!(
                r#"static_assert!({} : i32 where {} >= {} && {} <= {});"#,
                fn_format[1], fn_format[1], lim1, fn_format[1], lim2
            );
        }
    }
    if fn_name.starts_with("fix_right_shift_imm") {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        let lim = if fn_format[2] == "bits" {
            type_bits(in_t[1]).to_string()
        } else {
            fn_format[2].clone()
        };
        let fixed = if in_t[1].starts_with('u') {
            format!("return vdup{nself}(0);", nself = type_to_n_suffix(in_t[1]))
        } else {
            (lim.parse::<i32>().unwrap() - 1).to_string()
        };

        return format!(
            r#"let {name}: i32 = if {const_name} == {upper} {{ {fixed} }} else {{ N }};"#,
            name = fn_format[1].to_lowercase(),
            const_name = fn_format[1],
            upper = lim,
            fixed = fixed,
        );
    }

    if fn_name.starts_with("matchn") {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        let len = match &*fn_format[1] {
            "out_exp_len" => type_exp_len(out_t, 1),
            "in_exp_len" => type_exp_len(in_t[1], 1),
            "in0_exp_len" => type_exp_len(in_t[0], 1),
            _ => 0,
        };
        let mut call = format!("match {} & 0b{} {{\n", &fn_format[2], "1".repeat(len));
        let mut sub_call = String::new();
        for p in 1..params.len() {
            if !sub_call.is_empty() {
                sub_call.push_str(", ");
            }
            sub_call.push_str(&params[p]);
        }
        for i in 0..(2u32.pow(len as u32) as usize) {
            let sub_match = format!(
                "        {} => {},\n",
                i,
                get_call(
                    &sub_call,
                    current_name,
                    const_declare,
                    in_t,
                    out_t,
                    fixed,
                    Some(i as i32),
                    aarch64
                )
            );
            call.push_str(&sub_match);
        }
        call.push_str("        _ => unreachable_unchecked(),\n    }");
        return call;
    }
    let mut re: Option<(String, String)> = None;
    let mut param_str = String::new();
    let mut i = 1;
    while i < params.len() {
        let s = &params[i];
        if s.starts_with('{') {
            let mut sub_fn = String::new();
            let mut paranthes = 0;
            while i < params.len() {
                if !sub_fn.is_empty() {
                    sub_fn.push_str(", ");
                }
                sub_fn.push_str(&params[i]);
                let l = params[i].len();
                for j in 0..l {
                    if &params[i][j..j + 1] == "{" {
                        paranthes += 1;
                    } else {
                        break;
                    }
                }
                for j in 0..l {
                    if &params[i][l - j - 1..l - j] == "}" {
                        paranthes -= 1;
                    } else {
                        break;
                    }
                }
                if paranthes == 0 {
                    break;
                }
                i += 1;
            }
            let sub_call = get_call(
                &sub_fn[1..sub_fn.len() - 1],
                current_name,
                const_declare,
                in_t,
                out_t,
                fixed,
                n.clone(),
                aarch64,
            );
            if !param_str.is_empty() {
                param_str.push_str(", ");
            }
            param_str.push_str(&sub_call);
        } else if s.contains(':') {
            let re_params: Vec<_> = s.split(':').map(|v| v.to_string()).collect();
            if re_params[1] == "" {
                re = Some((re_params[0].clone(), in_t[1].to_string()));
            } else if re_params[1] == "in_t" {
                re = Some((re_params[0].clone(), in_t[1].to_string()));
            } else if re_params[1] == "signed" {
                re = Some((re_params[0].clone(), type_to_signed(in_t[1])));
            } else if re_params[1] == "unsigned" {
                re = Some((re_params[0].clone(), type_to_unsigned(in_t[1])));
            } else if re_params[1] == "in_t0" {
                re = Some((re_params[0].clone(), in_t[0].to_string()));
            } else if re_params[1] == "in_t1" {
                re = Some((re_params[0].clone(), in_t[1].to_string()));
            } else if re_params[1] == "out_t" {
                re = Some((re_params[0].clone(), out_t.to_string()));
            } else if re_params[1] == "half" {
                re = Some((re_params[0].clone(), type_to_half(in_t[1]).to_string()));
            } else if re_params[1] == "in_ntt" {
                re = Some((
                    re_params[0].clone(),
                    native_type_to_type(in_t[1]).to_string(),
                ));
            } else if re_params[1] == "in_long_ntt" {
                re = Some((
                    re_params[0].clone(),
                    native_type_to_long_type(in_t[1]).to_string(),
                ));
            } else if re_params[1] == "out_ntt" {
                re = Some((re_params[0].clone(), native_type_to_type(out_t).to_string()));
            } else if re_params[1] == "out_long_ntt" {
                re = Some((
                    re_params[0].clone(),
                    native_type_to_long_type(out_t).to_string(),
                ));
            } else {
                re = Some((re_params[0].clone(), re_params[1].clone()));
            }
        } else {
            if !param_str.is_empty() {
                param_str.push_str(", ");
            }
            param_str.push_str(s);
        }
        i += 1;
    }
    if fn_name == "fixed" {
        let (re_name, re_type) = re.unwrap();
        let fixed: Vec<String> = fixed.iter().take(type_len(in_t[1])).cloned().collect();
        return format!(r#"let {}{};"#, re_name, values(&re_type, &fixed));
    }
    if fn_name == "fixed-half-right" {
        let fixed: Vec<String> = fixed.iter().take(type_len(in_t[1])).cloned().collect();
        let half = fixed[type_len(in_t[1]) / 2..]
            .iter()
            .fold(String::new(), |mut s, fix| {
                s.push_str(fix);
                s.push_str(", ");
                s
            });
        return format!(r#"[{}]"#, &half[..half.len() - 2]);
    }
    if fn_name == "a - b" {
        return fn_name;
    }
    if fn_name == "-a" {
        return fn_name;
    }
    if fn_name.contains('-') {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        assert_eq!(fn_format.len(), 3);
        fn_name = if fn_format[0] == "self" {
            current_name.to_string()
        } else {
            fn_format[0].clone()
        };
        if fn_format[1] == "self" {
            fn_name.push_str(type_to_suffix(in_t[1]));
        } else if fn_format[1] == "nself" {
            fn_name.push_str(type_to_n_suffix(in_t[1]));
        } else if fn_format[1] == "nselfvfp4" {
            fn_name.push_str(type_to_n_suffix(in_t[1]));
            if !aarch64 {
                fn_name.push_str("_vfp4");
            }
        } else if fn_format[1] == "out" {
            fn_name.push_str(type_to_suffix(out_t));
        } else if fn_format[1] == "in0" {
            fn_name.push_str(type_to_suffix(in_t[0]));
        } else if fn_format[1] == "in2" {
            fn_name.push_str(type_to_suffix(in_t[2]));
        } else if fn_format[1] == "in2lane" {
            fn_name.push_str(&type_to_lane_suffixes(out_t, in_t[2], false));
        } else if fn_format[1] == "outlane" {
            fn_name.push_str(&type_to_lane_suffixes(out_t, in_t[2], true));
        } else if fn_format[1] == "signed" {
            fn_name.push_str(type_to_suffix(&type_to_signed(&String::from(in_t[1]))));
        } else if fn_format[1] == "outsigned" {
            fn_name.push_str(type_to_suffix(&type_to_signed(&String::from(out_t))));
        } else if fn_format[1] == "outsignednox" {
            fn_name.push_str(&type_to_suffix(&type_to_sub_type(&type_to_signed(
                &String::from(out_t),
            ))));
        } else if fn_format[1] == "in1signednox" {
            fn_name.push_str(&type_to_suffix(&type_to_sub_type(&type_to_signed(
                &String::from(in_t[1]),
            ))));
        } else if fn_format[1] == "outsigneddupnox" {
            fn_name.push_str(&type_to_dup_suffix(&type_to_sub_type(&type_to_signed(
                &String::from(out_t),
            ))));
        } else if fn_format[1] == "outsignedlanenox" {
            fn_name.push_str(&type_to_lane_suffix(&type_to_sub_type(&type_to_signed(
                &String::from(out_t),
            ))));
        } else if fn_format[1] == "in1signedlanenox" {
            fn_name.push_str(&type_to_lane_suffix(&type_to_sub_type(&type_to_signed(
                &String::from(in_t[1]),
            ))));
        } else if fn_format[1] == "unsigned" {
            fn_name.push_str(type_to_suffix(&type_to_unsigned(in_t[1])));
        } else if fn_format[1] == "doubleself" {
            fn_name.push_str(&type_to_double_suffixes(out_t, in_t[1]));
        } else if fn_format[1] == "noq_doubleself" {
            fn_name.push_str(&type_to_noq_double_suffixes(out_t, in_t[1]));
        } else if fn_format[1] == "noqself" {
            fn_name.push_str(type_to_noq_suffix(in_t[1]));
        } else if fn_format[1] == "noqsigned" {
            fn_name.push_str(type_to_noq_suffix(&type_to_signed(&String::from(in_t[1]))));
        } else if fn_format[1] == "nosuffix" {
        } else if fn_format[1] == "in_len" {
            fn_name.push_str(&type_len(in_t[1]).to_string());
        } else if fn_format[1] == "in0_len" {
            fn_name.push_str(&type_len(in_t[0]).to_string());
        } else if fn_format[1] == "out_len" {
            fn_name.push_str(&type_len(out_t).to_string());
        } else if fn_format[1] == "halflen" {
            fn_name.push_str(&(type_len(in_t[1]) / 2).to_string());
        } else if fn_format[1] == "nout" {
            fn_name.push_str(type_to_n_suffix(out_t));
        } else if fn_format[1] == "nin0" {
            fn_name.push_str(type_to_n_suffix(in_t[0]));
        } else if fn_format[1] == "nsigned" {
            fn_name.push_str(type_to_n_suffix(&type_to_signed(&String::from(in_t[1]))));
        } else if fn_format[1] == "in_ntt" {
            fn_name.push_str(type_to_suffix(native_type_to_type(in_t[1])));
        } else if fn_format[1] == "out_ntt" {
            fn_name.push_str(type_to_suffix(native_type_to_type(out_t)));
        } else if fn_format[1] == "rot" {
            fn_name = type_to_rot_suffix(&fn_name, type_to_suffix(out_t));
        } else {
            fn_name.push_str(&fn_format[1]);
        };
        if fn_format[2] == "ext" {
            fn_name.push_str("_");
        } else if fn_format[2] == "noext" {
        } else if fn_format[2].starts_with("<") {
            assert!(fn_format[2].ends_with(">"));
            let types: Vec<_> = fn_format[2][1..fn_format[2].len() - 1]
                .split(' ')
                .map(|v| v.to_string())
                .collect();
            assert_eq!(types.len(), 2);
            let type1 = if types[0] == "element_t" {
                type_to_native_type(in_t[1])
            } else {
                String::from(&types[0])
            };
            let type2 = if types[1] == "element_t" {
                type_to_native_type(in_t[1])
            } else {
                String::from(&types[1])
            };
            fn_name.push_str(&format!("::<{}, {}>", &type1, &type2));
        } else {
            fn_name.push_str(&fn_format[2]);
        }
    }
    if param_str.is_empty() {
        return fn_name.replace("out_t", out_t);
    }
    let fn_str = if let Some((re_name, re_type)) = re.clone() {
        format!(
            r#"let {}: {} = {}({});"#,
            re_name, re_type, fn_name, param_str
        )
    } else if fn_name.starts_with("*") {
        format!(r#"{} = {};"#, fn_name, param_str)
    } else {
        format!(r#"{}({})"#, fn_name, param_str)
    };
    return fn_str;
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let in_file = args.get(1).cloned().unwrap_or_else(|| IN.to_string());

    let f = File::open(in_file).expect("Failed to open neon.spec");
    let f = BufReader::new(f);

    let mut current_comment = String::new();
    let mut current_name: Option<String> = None;
    let mut current_fn: Option<String> = None;
    let mut current_arm: Option<String> = None;
    let mut current_aarch64: Option<String> = None;
    let mut link_arm: Option<String> = None;
    let mut link_aarch64: Option<String> = None;
    let mut const_arm: Option<String> = None;
    let mut const_aarch64: Option<String> = None;
    let mut constn: Option<String> = None;
    let mut para_num = 2;
    let mut suffix: Suffix = Normal;
    let mut a: Vec<String> = Vec::new();
    let mut b: Vec<String> = Vec::new();
    let mut c: Vec<String> = Vec::new();
    let mut n: Option<String> = None;
    let mut fixed: Vec<String> = Vec::new();
    let mut current_tests: Vec<(
        Vec<String>,
        Vec<String>,
        Vec<String>,
        Option<String>,
        Vec<String>,
    )> = Vec::new();
    let mut multi_fn: Vec<String> = Vec::new();
    let mut target: TargetFeature = Default;
    let mut fn_type: Fntype = Fntype::Normal;
    let mut separate = false;

    //
    // THIS FILE IS GENERATED FORM neon.spec DO NOT CHANGE IT MANUALLY
    //
    let mut out_arm = String::from(
        r#"// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen/neon.spec` and run the following command to re-generate this file:
//
// ```
// OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen -- crates/stdarch-gen/neon.spec
// ```
use super::*;
#[cfg(test)]
use stdarch_test::assert_instr;
"#,
    );
    let mut tests_arm = String::from(
        r#"
#[cfg(test)]
#[allow(overflowing_literals)]
mod test {
    use super::*;
    use crate::core_arch::simd::*;
    use std::mem::transmute;
    use stdarch_test::simd_test;
"#,
    );
    //
    // THIS FILE IS GENERATED FORM neon.spec DO NOT CHANGE IT MANUALLY
    //
    let mut out_aarch64 = String::from(
        r#"// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen/neon.spec` and run the following command to re-generate this file:
//
// ```
// OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen -- crates/stdarch-gen/neon.spec
// ```
use super::*;
#[cfg(test)]
use stdarch_test::assert_instr;
"#,
    );
    let mut tests_aarch64 = String::from(
        r#"
#[cfg(test)]
mod test {
    use super::*;
    use crate::core_arch::simd::*;
    use std::mem::transmute;
    use stdarch_test::simd_test;
"#,
    );

    for line in f.lines() {
        let line = line.unwrap();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("/// ") {
            current_comment = line;
            current_name = None;
            current_fn = None;
            current_arm = None;
            current_aarch64 = None;
            link_aarch64 = None;
            link_arm = None;
            const_aarch64 = None;
            const_arm = None;
            current_tests = Vec::new();
            constn = None;
            para_num = 2;
            suffix = Normal;
            a = Vec::new();
            b = Vec::new();
            c = Vec::new();
            fixed = Vec::new();
            n = None;
            multi_fn = Vec::new();
            target = Default;
            fn_type = Fntype::Normal;
            separate = false;
        } else if line.starts_with("//") {
        } else if line.starts_with("name = ") {
            current_name = Some(String::from(&line[7..]));
        } else if line.starts_with("fn = ") {
            current_fn = Some(String::from(&line[5..]));
        } else if line.starts_with("multi_fn = ") {
            multi_fn.push(String::from(&line[11..]));
        } else if line.starts_with("constn = ") {
            constn = Some(String::from(&line[9..]));
        } else if line.starts_with("arm = ") {
            current_arm = Some(String::from(&line[6..]));
        } else if line.starts_with("aarch64 = ") {
            current_aarch64 = Some(String::from(&line[10..]));
        } else if line.starts_with("double-suffixes") {
            suffix = Double;
        } else if line.starts_with("no-q") {
            suffix = NoQ;
        } else if line.starts_with("noq-double-suffixes") {
            suffix = NoQDouble;
        } else if line.starts_with("n-suffix") {
            suffix = NSuffix;
        } else if line.starts_with("double-n-suffixes") {
            suffix = DoubleN;
        } else if line.starts_with("out-n-suffix") {
            suffix = OutNSuffix;
        } else if line.starts_with("noq-n-suffix") {
            suffix = NoQNSuffix;
        } else if line.starts_with("out-suffix") {
            suffix = OutSuffix;
        } else if line.starts_with("out-nox") {
            suffix = OutNox;
        } else if line.starts_with("in1-nox") {
            suffix = In1Nox;
        } else if line.starts_with("out-dup-nox") {
            suffix = OutDupNox;
        } else if line.starts_with("out-lane-nox") {
            suffix = OutLaneNox;
        } else if line.starts_with("in1-lane-nox") {
            suffix = In1LaneNox;
        } else if line.starts_with("lane-suffixes") {
            suffix = Lane;
        } else if line.starts_with("in2-suffix") {
            suffix = In2;
        } else if line.starts_with("in2-lane-suffixes") {
            suffix = In2Lane;
        } else if line.starts_with("out-lane-suffixes") {
            suffix = OutLane;
        } else if line.starts_with("rot-suffix") {
            suffix = Rot;
        } else if line.starts_with("rot-lane-suffixes") {
            suffix = RotLane;
        } else if line.starts_with("a = ") {
            a = line[4..].split(',').map(|v| v.trim().to_string()).collect();
        } else if line.starts_with("b = ") {
            b = line[4..].split(',').map(|v| v.trim().to_string()).collect();
        } else if line.starts_with("c = ") {
            c = line[4..].split(',').map(|v| v.trim().to_string()).collect();
        } else if line.starts_with("n = ") {
            n = Some(String::from(&line[4..]));
        } else if line.starts_with("fixed = ") {
            fixed = line[8..].split(',').map(|v| v.trim().to_string()).collect();
        } else if line.starts_with("validate ") {
            let e = line[9..].split(',').map(|v| v.trim().to_string()).collect();
            current_tests.push((a.clone(), b.clone(), c.clone(), n.clone(), e));
        } else if line.starts_with("link-aarch64 = ") {
            link_aarch64 = Some(String::from(&line[15..]));
        } else if line.starts_with("const-aarch64 = ") {
            const_aarch64 = Some(String::from(&line[16..]));
        } else if line.starts_with("link-arm = ") {
            link_arm = Some(String::from(&line[11..]));
        } else if line.starts_with("const-arm = ") {
            const_arm = Some(String::from(&line[12..]));
        } else if line.starts_with("load_fn") {
            fn_type = Fntype::Load;
        } else if line.starts_with("store_fn") {
            fn_type = Fntype::Store;
        } else if line.starts_with("arm-aarch64-separate") {
            separate = true;
        } else if line.starts_with("target = ") {
            target = match Some(String::from(&line[9..])) {
                Some(input) => match input.as_str() {
                    "v7" => ArmV7,
                    "vfp4" => Vfp4,
                    "fp-armv8" => FPArmV8,
                    "aes" => AES,
                    "fcma" => FCMA,
                    "dotprod" => Dotprod,
                    "i8mm" => I8MM,
                    "sha3" => SHA3,
                    "rdm" => RDM,
                    "sm4" => SM4,
                    "frintts" => FTTS,
                    _ => Default,
                },
                _ => Default,
            }
        } else if line.starts_with("generate ") {
            let line = &line[9..];
            let types: Vec<String> = line
                .split(',')
                .map(|v| v.trim().to_string())
                .flat_map(|v| match v.as_str() {
                    "uint*_t" => UINT_TYPES.iter().map(|v| v.to_string()).collect(),
                    "uint64x*_t" => UINT_TYPES_64.iter().map(|v| v.to_string()).collect(),
                    "int*_t" => INT_TYPES.iter().map(|v| v.to_string()).collect(),
                    "int64x*_t" => INT_TYPES_64.iter().map(|v| v.to_string()).collect(),
                    "float*_t" => FLOAT_TYPES.iter().map(|v| v.to_string()).collect(),
                    "float64x*_t" => FLOAT_TYPES_64.iter().map(|v| v.to_string()).collect(),
                    _ => vec![v],
                })
                .collect();

            for line in types {
                let spec: Vec<&str> = line.split(':').map(|e| e.trim()).collect();
                let in_t: [&str; 3];
                let out_t;
                if spec.len() == 1 {
                    in_t = [spec[0], spec[0], spec[0]];
                    out_t = spec[0];
                } else if spec.len() == 2 {
                    in_t = [spec[0], spec[0], spec[0]];
                    out_t = spec[1];
                } else if spec.len() == 3 {
                    in_t = [spec[0], spec[1], spec[1]];
                    out_t = spec[2];
                } else if spec.len() == 4 {
                    in_t = [spec[0], spec[1], spec[2]];
                    out_t = spec[3];
                } else {
                    panic!("Bad spec: {}", line)
                }
                if b.len() == 0 {
                    if matches!(fn_type, Fntype::Store) {
                        para_num = 2;
                    } else {
                        para_num = 1;
                    }
                } else if c.len() != 0 {
                    para_num = 3;
                }
                let current_name = current_name.clone().unwrap();
                if let Some(current_arm) = current_arm.clone() {
                    let (function, test) = gen_arm(
                        &current_comment,
                        &current_fn,
                        &current_name,
                        &current_arm,
                        &link_arm,
                        &current_aarch64,
                        &link_aarch64,
                        &const_arm,
                        &const_aarch64,
                        &constn,
                        &in_t,
                        &out_t,
                        &current_tests,
                        suffix,
                        para_num,
                        target,
                        &fixed,
                        &multi_fn,
                        fn_type,
                        separate,
                    );
                    out_arm.push_str(&function);
                    tests_arm.push_str(&test);
                } else {
                    let (function, test) = gen_aarch64(
                        &current_comment,
                        &current_fn,
                        &current_name,
                        &current_aarch64,
                        &link_aarch64,
                        &const_aarch64,
                        &constn,
                        &in_t,
                        &out_t,
                        &current_tests,
                        suffix,
                        para_num,
                        target,
                        &fixed,
                        &multi_fn,
                        fn_type,
                    );
                    out_aarch64.push_str(&function);
                    tests_aarch64.push_str(&test);
                }
            }
        }
    }
    tests_arm.push('}');
    tests_arm.push('\n');
    tests_aarch64.push('}');
    tests_aarch64.push('\n');

    let arm_out_path: PathBuf =
        PathBuf::from(env::var("OUT_DIR").unwrap_or("crates/core_arch".to_string()))
            .join("src")
            .join("arm_shared")
            .join("neon");
    std::fs::create_dir_all(&arm_out_path)?;

    let mut file_arm = File::create(arm_out_path.join(ARM_OUT))?;
    file_arm.write_all(out_arm.as_bytes())?;
    file_arm.write_all(tests_arm.as_bytes())?;

    let aarch64_out_path: PathBuf =
        PathBuf::from(env::var("OUT_DIR").unwrap_or("crates/core_arch".to_string()))
            .join("src")
            .join("aarch64")
            .join("neon");
    std::fs::create_dir_all(&aarch64_out_path)?;

    let mut file_aarch = File::create(aarch64_out_path.join(AARCH64_OUT))?;
    file_aarch.write_all(out_aarch64.as_bytes())?;
    file_aarch.write_all(tests_aarch64.as_bytes())?;
    /*
    if let Err(e) = Command::new("rustfmt")
        .arg(&arm_out_path)
        .arg(&aarch64_out_path)
        .status() {
            eprintln!("Could not format `{}`: {}", arm_out_path.to_str().unwrap(), e);
            eprintln!("Could not format `{}`: {}", aarch64_out_path.to_str().unwrap(), e);
    };
    */
    Ok(())
}
