use self::Suffix::*;
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
    match t {
        "int8x8_t" => 8,
        "int8x16_t" => 16,
        "int16x4_t" => 4,
        "int16x8_t" => 8,
        "int32x2_t" => 2,
        "int32x4_t" => 4,
        "int64x1_t" => 1,
        "int64x2_t" => 2,
        "uint8x8_t" => 8,
        "uint8x16_t" => 16,
        "uint16x4_t" => 4,
        "uint16x8_t" => 8,
        "uint32x2_t" => 2,
        "uint32x4_t" => 4,
        "uint64x1_t" => 1,
        "uint64x2_t" => 2,
        "float16x4_t" => 4,
        "float16x8_t" => 8,
        "float32x2_t" => 2,
        "float32x4_t" => 4,
        "float64x1_t" => 1,
        "float64x2_t" => 2,
        "poly8x8_t" => 8,
        "poly8x16_t" => 16,
        "poly16x4_t" => 4,
        "poly16x8_t" => 8,
        "poly64x1_t" => 1,
        "poly64x2_t" => 2,
        _ => panic!("unknown type: {}", t),
    }
}

fn type_exp_len(t: &str) -> usize {
    match t {
        "int8x8_t" => 3,
        "int8x16_t" => 4,
        "int16x4_t" => 2,
        "int16x8_t" => 3,
        "int32x2_t" => 1,
        "int32x4_t" => 2,
        "int64x1_t" => 0,
        "int64x2_t" => 1,
        "uint8x8_t" => 3,
        "uint8x16_t" => 4,
        "uint16x4_t" => 2,
        "uint16x8_t" => 3,
        "uint32x2_t" => 1,
        "uint32x4_t" => 2,
        "uint64x1_t" => 0,
        "uint64x2_t" => 1,
        "float16x4_t" => 2,
        "float16x8_t" => 3,
        "float32x2_t" => 1,
        "float32x4_t" => 2,
        "float64x1_t" => 0,
        "float64x2_t" => 1,
        "poly8x8_t" => 3,
        "poly8x16_t" => 4,
        "poly16x4_t" => 2,
        "poly16x8_t" => 3,
        "poly64x1_t" => 0,
        "poly64x2_t" => 1,
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
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_signed_suffix(t: &str) -> &str {
    match t {
        "int8x8_t" | "uint8x8_t" | "poly8x8_t" => "_s8",
        "int8x16_t" | "uint8x16_t" | "poly8x16_t" => "q_s8",
        "int16x4_t" | "uint16x4_t" | "poly16x4_t" => "_s16",
        "int16x8_t" | "uint16x8_t" | "poly16x8_t" => "q_s16",
        "int32x2_t" | "uint32x2_t" => "_s32",
        "int32x4_t" | "uint32x4_t" => "q_s32",
        "int64x1_t" | "uint64x1_t" | "poly64x1_t" => "_s64",
        "int64x2_t" | "uint64x2_t" | "poly64x2_t" => "q_s64",
        /*
        "float16x4_t" => "_f16",
        "float16x8_t" => "q_f16",
        "float32x2_t" => "_f32",
        "float32x4_t" => "q_f32",
        "float64x1_t" => "_f64",
        "float64x2_t" => "q_f64",
        "poly64x1_t" => "_p64",
        "poly64x2_t" => "q_p64",
         */
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_unsigned_suffix(t: &str) -> &str {
    match t {
        "int8x8_t" | "uint8x8_t" => "_u8",
        "int8x16_t" | "uint8x16_t" => "q_u8",
        "int16x4_t" | "uint16x4_t" => "_u16",
        "int16x8_t" | "uint16x8_t" => "q_u16",
        "int32x2_t" | "uint32x2_t" => "_u32",
        "int32x4_t" | "uint32x4_t" => "q_u32",
        "int64x1_t" | "uint64x1_t" => "_u64",
        "int64x2_t" | "uint64x2_t" => "q_u64",
        /*
        "float16x4_t" => "_f16",
        "float16x8_t" => "q_f16",
        "float32x2_t" => "_f32",
        "float32x4_t" => "q_f32",
        "float64x1_t" => "_f64",
        "float64x2_t" => "q_f64",
        "poly64x1_t" => "_p64",
        "poly64x2_t" => "q_p64",
         */
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_double_suffixes<'a>(out_t: &'a str, in_t: &'a str) -> String {
    let mut str = String::new();
    if type_to_suffix(in_t).starts_with("q") && type_to_suffix(out_t).starts_with("q") {
        str.push_str("q");
    }
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
        "int8x8_t" | "int8x16_t" => "_s8",
        "int16x4_t" | "int16x8_t" => "_s16",
        "int32x2_t" | "int32x4_t" => "_s32",
        "int64x1_t" | "int64x2_t" => "_s64",
        "uint8x8_t" | "uint8x16_t" => "_u8",
        "uint16x4_t" | "uint16x8_t" => "_u16",
        "uint32x2_t" | "uint32x4_t" => "_u32",
        "uint64x1_t" | "uint64x2_t" => "_u64",
        "float16x4_t" | "float16x8_t" => "_f16",
        "float32x2_t" | "float32x4_t" => "_f32",
        "float64x1_t" | "float64x2_t" => "_f64",
        "poly8x8_t" | "poly8x16_t" => "_p8",
        "poly16x4_t" | "poly16x8_t" => "_p16",
        "poly64x1_t" | "poly64x2_t" => "_p64",
        _ => panic!("unknown type: {}", t),
    }
}

#[derive(Clone, Copy)]
enum Suffix {
    Normal,
    Double,
    NoQ,
    NoQDouble,
}

fn type_to_global_type(t: &str) -> &str {
    match t {
        "int8x8_t" => "i8x8",
        "int8x16_t" => "i8x16",
        "int16x4_t" => "i16x4",
        "int16x8_t" => "i16x8",
        "int32x2_t" => "i32x2",
        "int32x4_t" => "i32x4",
        "int64x1_t" => "i64x1",
        "int64x2_t" => "i64x2",
        "uint8x8_t" => "u8x8",
        "uint8x16_t" => "u8x16",
        "uint16x4_t" => "u16x4",
        "uint16x8_t" => "u16x8",
        "uint32x2_t" => "u32x2",
        "uint32x4_t" => "u32x4",
        "uint64x1_t" => "u64x1",
        "uint64x2_t" => "u64x2",
        "float16x4_t" => "f16x4",
        "float16x8_t" => "f16x8",
        "float32x2_t" => "f32x2",
        "float32x4_t" => "f32x4",
        "float64x1_t" => "f64",
        "float64x2_t" => "f64x2",
        "poly8x8_t" => "i8x8",
        "poly8x16_t" => "i8x16",
        "poly16x4_t" => "i16x4",
        "poly16x8_t" => "i16x8",
        "poly64x1_t" => "i64x1",
        "poly64x2_t" => "i64x2",
        _ => panic!("unknown type: {}", t),
    }
}

// fn type_to_native_type(t: &str) -> &str {
//     match t {
//         "int8x8_t" => "i8",
//         "int8x16_t" => "i8",
//         "int16x4_t" => "i16",
//         "int16x8_t" => "i16",
//         "int32x2_t" => "i32",
//         "int32x4_t" => "i32",
//         "int64x1_t" => "i64",
//         "int64x2_t" => "i64",
//         "uint8x8_t" => "u8",
//         "uint8x16_t" => "u8",
//         "uint16x4_t" => "u16",
//         "uint16x8_t" => "u16",
//         "uint32x2_t" => "u32",
//         "uint32x4_t" => "u32",
//         "uint64x1_t" => "u64",
//         "uint64x2_t" => "u64",
//         "float16x4_t" => "f16",
//         "float16x8_t" => "f16",
//         "float32x2_t" => "f32",
//         "float32x4_t" => "f32",
//         "float64x1_t" => "f64",
//         "float64x2_t" => "f64",
//         "poly64x1_t" => "i64",
//         "poly64x2_t" => "i64",
//         _ => panic!("unknown type: {}", t),
//     }
// }

fn type_to_ext(t: &str) -> &str {
    match t {
        "int8x8_t" => "v8i8",
        "int8x16_t" => "v16i8",
        "int16x4_t" => "v4i16",
        "int16x8_t" => "v8i16",
        "int32x2_t" => "v2i32",
        "int32x4_t" => "v4i32",
        "int64x1_t" => "v1i64",
        "int64x2_t" => "v2i64",
        "uint8x8_t" => "v8i8",
        "uint8x16_t" => "v16i8",
        "uint16x4_t" => "v4i16",
        "uint16x8_t" => "v8i16",
        "uint32x2_t" => "v2i32",
        "uint32x4_t" => "v4i32",
        "uint64x1_t" => "v1i64",
        "uint64x2_t" => "v2i64",
        "float16x4_t" => "v4f16",
        "float16x8_t" => "v8f16",
        "float32x2_t" => "v2f32",
        "float32x4_t" => "v4f32",
        "float64x1_t" => "v1f64",
        "float64x2_t" => "v2f64",
        "poly8x8_t" => "v8i8",
        "poly8x16_t" => "v16i8",
        "poly16x4_t" => "v4i16",
        "poly16x8_t" => "v8i16",
        /*
        "poly64x1_t" => "i64x1",
        "poly64x2_t" => "i64x2",
        */
        _ => panic!("unknown type for extension: {}", t),
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

fn values(t: &str, vs: &[String]) -> String {
    if vs.len() == 1 && !t.contains('x') {
        format!(": {} = {}", t, vs[0])
    } else if vs.len() == 1 && type_to_global_type(t) == "f64" {
        format!(": {} = {}", type_to_global_type(t), vs[0])
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

#[allow(clippy::too_many_arguments)]
fn gen_aarch64(
    current_comment: &str,
    current_fn: &Option<String>,
    current_name: &str,
    current_aarch64: &Option<String>,
    link_aarch64: &Option<String>,
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
    fixed: &Vec<String>,
    multi_fn: &Vec<String>,
) -> (String, String) {
    let _global_t = type_to_global_type(in_t[0]);
    let _global_ret_t = type_to_global_type(out_t);
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
    };
    let current_fn = if let Some(current_fn) = current_fn.clone() {
        if link_aarch64.is_some() {
            panic!(
                "[{}] Can't specify link and (multi) fn at the same time.",
                name
            )
        }
        current_fn
    } else if !multi_fn.is_empty() {
        if link_aarch64.is_some() {
            panic!(
                "[{}] Can't specify link and (multi) fn at the same time.",
                name
            )
        }
        String::new()
    } else {
        if link_aarch64.is_none() {
            panic!(
                "[{}] Either (multi) fn or link-aarch have to be specified.",
                name
            )
        }
        format!("{}_", name)
    };
    let current_aarch64 = current_aarch64.clone().unwrap();
    let ext_c = if let Some(link_aarch64) = link_aarch64.clone() {
        let ext = type_to_ext(in_t[0]);
        let ext2 = type_to_ext(out_t);
        let link_aarch64 = if link_aarch64.starts_with("llvm") {
            link_aarch64.replace("_EXT_", ext).replace("_EXT2_", ext2)
        } else {
            let mut link = String::from("llvm.aarch64.neon.");
            link.push_str(&link_aarch64);
            link.replace("_EXT_", ext).replace("_EXT2_", ext2)
        };
        format!(
            r#"#[allow(improper_ctypes)]
    extern "C" {{
        #[cfg_attr(target_arch = "aarch64", link_name = "{}")]
        fn {}({}) -> {};
    }}
    "#,
            link_aarch64,
            current_fn,
            match para_num {
                1 => {
                    format!("a: {}", in_t[0])
                }
                2 => {
                    format!("a: {}, b: {}", in_t[0], in_t[1])
                }
                3 => {
                    format!("a: {}, b: {}, c: {}", in_t[0], in_t[1], in_t[2])
                }
                _ => unimplemented!("unknown para_num"),
            },
            out_t
        )
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
                in_t,
                out_t,
                fixed,
                None,
            ));
        }
        calls
    } else {
        String::new()
    };
    let const_declare = if let Some(constn) = constn {
        format!(r#"<const {}: i32>"#, constn)
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
    let call = match (multi_calls.len(), para_num, fixed.len()) {
        (0, 1, 0) => format!(
            r#"pub unsafe fn {}{}(a: {}) -> {} {{
    {}{}(a)
}}"#,
            name, const_declare, in_t[0], out_t, ext_c, current_fn,
        ),
        (0, 1, _) => {
            let fixed: Vec<String> = fixed.iter().take(type_len(in_t[0])).cloned().collect();
            format!(
                r#"pub unsafe fn {}{}(a: {}) -> {} {{
    let b{};
    {}{}(a, transmute(b))
}}"#,
                name,
                const_declare,
                in_t[0],
                out_t,
                values(in_t[0], &fixed),
                ext_c,
                current_fn,
            )
        }
        (0, 2, _) => format!(
            r#"pub unsafe fn {}{}(a: {}, b: {}) -> {} {{
    {}{}(a, b)
}}"#,
            name, const_declare, in_t[0], in_t[1], out_t, ext_c, current_fn,
        ),
        (0, 3, _) => format!(
            r#"pub unsafe fn {}{}(a: {}, b: {}, c: {}) -> {} {{
    {}{}(a, b, c)
}}"#,
            name, const_declare, in_t[0], in_t[1], in_t[2], out_t, ext_c, current_fn,
        ),
        (_, 1, _) => format!(
            r#"pub unsafe fn {}{}(a: {}) -> {} {{
    {}{}
}}"#,
            name, const_declare, in_t[0], out_t, ext_c, multi_calls,
        ),
        (_, 2, _) => format!(
            r#"pub unsafe fn {}{}(a: {}, b: {}) -> {} {{
    {}{}
}}"#,
            name, const_declare, in_t[0], in_t[1], out_t, ext_c, multi_calls,
        ),
        (_, 3, _) => format!(
            r#"pub unsafe fn {}{}(a: {}, b: {}, c: {}) -> {} {{
    {}{}
}}"#,
            name, const_declare, in_t[0], in_t[1], in_t[2], out_t, ext_c, multi_calls,
        ),
        (_, _, _) => String::new(),
    };
    let function = format!(
        r#"
{}
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr({}{}))]{}
{}
"#,
        current_comment, current_aarch64, const_assert, const_legacy, call
    );

    let test = gen_test(
        &name,
        in_t,
        &out_t,
        current_tests,
        [type_len(in_t[0]), type_len(in_t[1]), type_len(in_t[2])],
        type_len(out_t),
        para_num,
    );
    (function, test)
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
) -> String {
    let mut test = format!(
        r#"
    #[simd_test(enable = "neon")]
    unsafe fn test_{}() {{"#,
        name,
    );
    for (a, b, c, n, e) in current_tests {
        let a: Vec<String> = a.iter().take(len_in[0]).cloned().collect();
        let b: Vec<String> = b.iter().take(len_in[1]).cloned().collect();
        let c: Vec<String> = c.iter().take(len_in[2]).cloned().collect();
        let e: Vec<String> = e.iter().take(len_out).cloned().collect();
        let const_value = if let Some(constn) = n {
            format!(r#"::<{}>"#, map_val(in_t[1], constn))
        } else {
            String::new()
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
                        type_to_global_type(out_t),
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
                        type_to_global_type(out_t),
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
                        type_to_global_type(out_t),
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
    fixed: &Vec<String>,
    multi_fn: &Vec<String>,
) -> (String, String) {
    let _global_t = type_to_global_type(in_t[0]);
    let _global_ret_t = type_to_global_type(out_t);
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
    };
    let current_aarch64 = current_aarch64
        .clone()
        .unwrap_or_else(|| current_arm.to_string());

    let current_fn = if let Some(current_fn) = current_fn.clone() {
        if link_aarch64.is_some() || link_arm.is_some() {
            panic!(
                "[{}] Can't specify link and function at the same time. {} / {:?} / {:?}",
                name, current_fn, link_aarch64, link_arm
            )
        }
        current_fn
    } else if !multi_fn.is_empty() {
        if link_aarch64.is_some() || link_arm.is_some() {
            panic!(
                "[{}] Can't specify link and function at the same time. multi_fn / {:?} / {:?}",
                name, link_aarch64, link_arm
            )
        }
        String::new()
    } else {
        if link_aarch64.is_none() && link_arm.is_none() {
            panic!(
                "[{}] Either fn or link-arm and link-aarch have to be specified.",
                name
            )
        }
        format!("{}_", name)
    };
    let ext_c =
        if let (Some(link_arm), Some(link_aarch64)) = (link_arm.clone(), link_aarch64.clone()) {
            let ext = type_to_ext(in_t[0]);
            let ext2 = type_to_ext(out_t);
            let link_arm = if link_arm.starts_with("llvm") {
                link_arm.replace("_EXT_", ext).replace("_EXT2_", ext2)
            } else {
                let mut link = String::from("llvm.arm.neon.");
                link.push_str(&link_arm);
                link.replace("_EXT_", ext).replace("_EXT2_", ext2)
            };
            let link_aarch64 = if link_aarch64.starts_with("llvm") {
                link_aarch64.replace("_EXT_", ext).replace("_EXT2_", ext2)
            } else {
                let mut link = String::from("llvm.aarch64.neon.");
                link.push_str(&link_aarch64);
                link.replace("_EXT_", ext).replace("_EXT2_", ext2)
            };
            format!(
                r#"#[allow(improper_ctypes)]
    extern "C" {{
        #[cfg_attr(target_arch = "arm", link_name = "{}")]
        #[cfg_attr(target_arch = "aarch64", link_name = "{}")]
        fn {}({}) -> {};
    }}
"#,
                link_arm,
                link_aarch64,
                current_fn,
                match para_num {
                    1 => {
                        format!("a: {}", in_t[0])
                    }
                    2 => {
                        format!("a: {}, b: {}", in_t[0], in_t[1])
                    }
                    3 => {
                        format!("a: {}, b: {}, c: {}", in_t[0], in_t[1], in_t[2])
                    }
                    _ => unimplemented!("unknown para_num"),
                },
                out_t
            )
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
                in_t,
                out_t,
                fixed,
                None,
            ));
        }
        calls
    } else {
        String::new()
    };
    let const_declare = if let Some(constn) = constn {
        format!(r#"<const {}: i32>"#, constn)
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
    let call = match (multi_calls.len(), para_num, fixed.len()) {
        (0, 1, 0) => format!(
            r#"pub unsafe fn {}{}(a: {}) -> {} {{
    {}{}(a)
}}"#,
            name, const_declare, in_t[0], out_t, ext_c, current_fn,
        ),
        (0, 1, _) => {
            let fixed: Vec<String> = fixed.iter().take(type_len(in_t[0])).cloned().collect();
            format!(
                r#"pub unsafe fn {}{}(a: {}) -> {} {{
    let b{};
    {}{}(a, transmute(b))
}}"#,
                name,
                const_declare,
                in_t[0],
                out_t,
                values(in_t[0], &fixed),
                ext_c,
                current_fn,
            )
        }
        (0, 2, _) => format!(
            r#"pub unsafe fn {}{}(a: {}, b: {}) -> {} {{
    {}{}(a, b)
}}"#,
            name, const_declare, in_t[0], in_t[1], out_t, ext_c, current_fn,
        ),
        (0, 3, _) => format!(
            r#"pub unsafe fn {}{}(a: {}, b: {}, c: {}) -> {} {{
    {}{}(a, b)
}}"#,
            name, const_declare, in_t[0], in_t[1], in_t[2], out_t, ext_c, current_fn,
        ),
        (_, 1, _) => format!(
            r#"pub unsafe fn {}{}(a: {}) -> {} {{
    {}{}
}}"#,
            name, const_declare, in_t[0], out_t, ext_c, multi_calls,
        ),
        (_, 2, _) => format!(
            r#"pub unsafe fn {}{}(a: {}, b: {}) -> {} {{
    {}{}
}}"#,
            name, const_declare, in_t[0], in_t[1], out_t, ext_c, multi_calls,
        ),
        (_, 3, _) => format!(
            r#"pub unsafe fn {}{}(a: {}, b: {}, c: {}) -> {} {{
    {}{}
}}"#,
            name, const_declare, in_t[0], in_t[1], in_t[2], out_t, ext_c, multi_calls,
        ),
        (_, _, _) => String::new(),
    };
    let function = format!(
        r#"
{}
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr({}{}))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr({}{}))]{}
{}
"#,
        current_comment,
        expand_intrinsic(&current_arm, in_t[1]),
        const_assert,
        expand_intrinsic(&current_aarch64, in_t[1]),
        const_assert,
        const_legacy,
        call,
    );
    let test = gen_test(
        &name,
        in_t,
        &out_t,
        current_tests,
        [type_len(in_t[0]), type_len(in_t[1]), type_len(in_t[2])],
        type_len(out_t),
        para_num,
    );

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
    } else {
        intr.to_string()
    }
}

fn get_call(
    in_str: &str,
    current_name: &str,
    in_t: &[&str; 3],
    out_t: &str,
    fixed: &Vec<String>,
    n: Option<i32>,
) -> String {
    let params: Vec<_> = in_str.split(',').map(|v| v.trim().to_string()).collect();
    assert!(params.len() > 0);
    let mut fn_name = params[0].clone();
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
    if fn_name.starts_with("asc") {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        let start = match &*fn_format[1] {
            "0" => 0,
            "n" => n.unwrap(),
            s => s.parse::<i32>().unwrap(),
        };
        let len = match &*fn_format[2] {
            "out_len" => type_len(out_t),
            "in_len" => type_len(in_t[1]),
            _ => 0,
        };
        return asc(start, len);
    }
    if fn_name.starts_with("static_assert_imm") {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        let len = match &*fn_format[1] {
            "out_exp_len" => type_exp_len(out_t),
            "in_exp_len" => type_exp_len(in_t[1]),
            _ => 0,
        };
        let sa = format!(r#"static_assert_imm{}!({});"#, len, fn_format[2]);
        return sa;
    }
    if fn_name.starts_with("matchn") {
        let fn_format: Vec<_> = fn_name.split('-').map(|v| v.to_string()).collect();
        let len = match &*fn_format[1] {
            "out_exp_len" => type_exp_len(out_t),
            "in_exp_len" => type_exp_len(in_t[1]),
            _ => 0,
        };
        let mut call = format!("match N & 0b{} {{\n", "1".repeat(len));
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
                get_call(&sub_call, current_name, in_t, out_t, fixed, Some(i as i32))
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
                if params[i].starts_with('{') {
                    paranthes += 1;
                }
                if params[i].ends_with('}') {
                    paranthes -= 1;
                    if paranthes == 0 {
                        break;
                    }
                }
                i += 1;
            }
            let sub_call = get_call(
                &sub_fn[1..sub_fn.len() - 1],
                current_name,
                in_t,
                out_t,
                fixed,
                n.clone(),
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
            } else if re_params[1] == "in_t0" {
                re = Some((re_params[0].clone(), in_t[0].to_string()));
            } else if re_params[1] == "in_t1" {
                re = Some((re_params[0].clone(), in_t[1].to_string()));
            } else if re_params[1] == "out_t" {
                re = Some((re_params[0].clone(), out_t.to_string()));
            } else if re_params[1] == "half" {
                re = Some((re_params[0].clone(), type_to_half(in_t[1]).to_string()));
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
        } else if fn_format[1] == "signed" {
            fn_name.push_str(type_to_signed_suffix(in_t[1]));
        } else if fn_format[1] == "unsigned" {
            fn_name.push_str(type_to_unsigned_suffix(in_t[1]));
        } else if fn_format[1] == "doubleself" {
            fn_name.push_str(&type_to_double_suffixes(out_t, in_t[1]));
        } else if fn_format[1] == "noq_doubleself" {
            fn_name.push_str(&type_to_noq_double_suffixes(out_t, in_t[1]));
        } else if fn_format[1] == "noqself" {
            fn_name.push_str(type_to_noq_suffix(in_t[1]));
        } else if fn_format[1] == "nosuffix" {
        } else if fn_format[1] == "in_len" {
            fn_name.push_str(&type_len(in_t[1]).to_string());
        } else if fn_format[1] == "out_len" {
            fn_name.push_str(&type_len(out_t).to_string());
        } else {
            fn_name.push_str(&fn_format[1]);
        };
        if fn_format[2] == "ext" {
            fn_name.push_str("_");
        }
    }
    if param_str.is_empty() {
        param_str.push_str("a, b");
    }
    let fn_str = if let Some((re_name, re_type)) = re.clone() {
        format!(
            r#"let {}: {} = {}({});"#,
            re_name, re_type, fn_name, param_str
        )
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
        } else if line.starts_with("link-arm = ") {
            link_arm = Some(String::from(&line[11..]));
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
                    para_num = 1;
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
                        &constn,
                        &in_t,
                        &out_t,
                        &current_tests,
                        suffix,
                        para_num,
                        &fixed,
                        &multi_fn,
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
                        &constn,
                        &in_t,
                        &out_t,
                        &current_tests,
                        suffix,
                        para_num,
                        &fixed,
                        &multi_fn,
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

    let arm_out_path: PathBuf = PathBuf::from(env::var("OUT_DIR").unwrap())
        .join("src")
        .join("arm")
        .join("neon");
    std::fs::create_dir_all(&arm_out_path)?;

    let mut file_arm = File::create(arm_out_path.join(ARM_OUT))?;
    file_arm.write_all(out_arm.as_bytes())?;
    file_arm.write_all(tests_arm.as_bytes())?;

    let aarch64_out_path: PathBuf = PathBuf::from(env::var("OUT_DIR").unwrap())
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
