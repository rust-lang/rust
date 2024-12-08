#![allow(rustc::internal)]

use std::fmt::Debug;
use std::fs;

use rustc_macros::{Decodable_Generic, Encodable_Generic};
use rustc_serialize::opaque::{FileEncoder, MemDecoder};
use rustc_serialize::{Decodable, Encodable};

#[derive(PartialEq, Clone, Debug, Encodable_Generic, Decodable_Generic)]
struct Struct {
    a: (),
    b: u8,
    c: u16,
    d: u32,
    e: u64,
    f: usize,

    g: i8,
    h: i16,
    i: i32,
    j: i64,
    k: isize,

    l: char,
    m: String,
    p: bool,
    q: Option<u32>,
}

fn check_round_trip<
    T: Encodable<FileEncoder> + for<'a> Decodable<MemDecoder<'a>> + PartialEq + Debug,
>(
    values: Vec<T>,
) {
    let tmpfile = tempfile::NamedTempFile::new().unwrap();
    let tmpfile = tmpfile.path();

    let mut encoder = FileEncoder::new(&tmpfile).unwrap();
    for value in &values {
        Encodable::encode(value, &mut encoder);
    }
    encoder.finish().unwrap();

    let data = fs::read(&tmpfile).unwrap();
    let mut decoder = MemDecoder::new(&data[..], 0).unwrap();
    for value in values {
        let decoded = Decodable::decode(&mut decoder);
        assert_eq!(value, decoded);
    }
}

#[test]
fn test_unit() {
    check_round_trip(vec![(), (), (), ()]);
}

#[test]
fn test_u8() {
    let mut vec = vec![];
    for i in u8::MIN..u8::MAX {
        vec.push(i);
    }
    check_round_trip(vec);
}

#[test]
fn test_u16() {
    for i in [u16::MIN, 111, 3333, 55555, u16::MAX] {
        check_round_trip(vec![1, 2, 3, i, i, i]);
    }
}

#[test]
fn test_u32() {
    check_round_trip(vec![1, 2, 3, u32::MIN, 0, 1, u32::MAX, 2, 1]);
}

#[test]
fn test_u64() {
    check_round_trip(vec![1, 2, 3, u64::MIN, 0, 1, u64::MAX, 2, 1]);
}

#[test]
fn test_usize() {
    check_round_trip(vec![1, 2, 3, usize::MIN, 0, 1, usize::MAX, 2, 1]);
}

#[test]
fn test_i8() {
    let mut vec = vec![];
    for i in i8::MIN..i8::MAX {
        vec.push(i);
    }
    check_round_trip(vec);
}

#[test]
fn test_i16() {
    for i in [i16::MIN, -100, 0, 101, i16::MAX] {
        check_round_trip(vec![-1, 2, -3, i, i, i, 2]);
    }
}

#[test]
fn test_i32() {
    check_round_trip(vec![-1, 2, -3, i32::MIN, 0, 1, i32::MAX, 2, 1]);
}

#[test]
fn test_i64() {
    check_round_trip(vec![-1, 2, -3, i64::MIN, 0, 1, i64::MAX, 2, 1]);
}

#[test]
fn test_isize() {
    check_round_trip(vec![-1, 2, -3, isize::MIN, 0, 1, isize::MAX, 2, 1]);
}

#[test]
fn test_bool() {
    check_round_trip(vec![false, true, true, false, false]);
}

#[test]
fn test_char() {
    let vec = vec!['a', 'b', 'c', 'd', 'A', 'X', ' ', '#', 'Ö', 'Ä', 'µ', '€'];
    check_round_trip(vec);
}

#[test]
fn test_string() {
    let vec = vec![
        "abcbuÖeiovÄnameÜavmpßvmea€µsbpnvapeapmaebn".to_string(),
        "abcbuÖganeiovÄnameÜavmpßvmea€µsbpnvapeapmaebn".to_string(),
        "abcbuÖganeiovÄnameÜavmpßvmea€µsbpapmaebn".to_string(),
        "abcbuÖganeiovÄnameÜavmpßvmeabpnvapeapmaebn".to_string(),
        "abcbuÖganeiÄnameÜavmpßvmea€µsbpnvapeapmaebn".to_string(),
        "abcbuÖganeiovÄnameÜavmpßvmea€µsbpmaebn".to_string(),
        "abcbuÖganeiovÄnameÜavmpßvmea€µnvapeapmaebn".to_string(),
    ];

    check_round_trip(vec);
}

#[test]
fn test_option() {
    check_round_trip(vec![Some(-1i8)]);
    check_round_trip(vec![Some(-2i16)]);
    check_round_trip(vec![Some(-3i32)]);
    check_round_trip(vec![Some(-4i64)]);
    check_round_trip(vec![Some(-5isize)]);

    let none_i8: Option<i8> = None;
    check_round_trip(vec![none_i8]);

    let none_i16: Option<i16> = None;
    check_round_trip(vec![none_i16]);

    let none_i32: Option<i32> = None;
    check_round_trip(vec![none_i32]);

    let none_i64: Option<i64> = None;
    check_round_trip(vec![none_i64]);

    let none_isize: Option<isize> = None;
    check_round_trip(vec![none_isize]);
}

#[test]
fn test_struct() {
    check_round_trip(vec![Struct {
        a: (),
        b: 10,
        c: 11,
        d: 12,
        e: 13,
        f: 14,

        g: 15,
        h: 16,
        i: 17,
        j: 18,
        k: 19,

        l: 'x',
        m: "abc".to_string(),
        p: false,
        q: None,
    }]);

    check_round_trip(vec![Struct {
        a: (),
        b: 101,
        c: 111,
        d: 121,
        e: 131,
        f: 141,

        g: -15,
        h: -16,
        i: -17,
        j: -18,
        k: -19,

        l: 'y',
        m: "def".to_string(),
        p: true,
        q: Some(1234567),
    }]);
}

#[derive(PartialEq, Clone, Debug, Encodable_Generic, Decodable_Generic)]
enum Enum {
    Variant1,
    Variant2(usize, u32),
    Variant3 { a: i32, b: char, c: bool },
}

#[test]
fn test_enum() {
    check_round_trip(vec![
        Enum::Variant1,
        Enum::Variant2(1, 25),
        Enum::Variant3 { a: 3, b: 'b', c: false },
        Enum::Variant3 { a: -4, b: 'f', c: true },
    ]);
}

#[test]
fn test_sequence() {
    let mut vec = vec![];
    for i in -100i64..100i64 {
        vec.push(i * 100000);
    }

    check_round_trip(vec![vec]);
}

#[test]
fn test_hash_map() {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    for i in -100i64..100i64 {
        map.insert(i * 100000, i * 10000);
    }

    check_round_trip(vec![map]);
}

#[test]
fn test_tuples() {
    check_round_trip(vec![('x', (), false, 5u32)]);
    check_round_trip(vec![(9i8, 10u16, 15i64)]);
    check_round_trip(vec![(-12i16, 11u8, 12usize)]);
    check_round_trip(vec![(1234567isize, 100000000000000u64, 99999999999999i64)]);
    check_round_trip(vec![(String::new(), "some string".to_string())]);
}

#[test]
fn test_unit_like_struct() {
    #[derive(Encodable_Generic, Decodable_Generic, PartialEq, Debug)]
    struct UnitLikeStruct;

    check_round_trip(vec![UnitLikeStruct]);
}

#[test]
fn test_box() {
    #[derive(Encodable_Generic, Decodable_Generic, PartialEq, Debug)]
    struct A {
        foo: Box<[bool]>,
    }

    let obj = A { foo: Box::new([true, false]) };
    check_round_trip(vec![obj]);
}

#[test]
fn test_cell() {
    use std::cell::{Cell, RefCell};

    #[derive(Encodable_Generic, Decodable_Generic, PartialEq, Debug)]
    struct A {
        baz: isize,
    }

    #[derive(Encodable_Generic, Decodable_Generic, PartialEq, Debug)]
    struct B {
        foo: Cell<bool>,
        bar: RefCell<A>,
    }

    let obj = B { foo: Cell::new(true), bar: RefCell::new(A { baz: 2 }) };
    check_round_trip(vec![obj]);
}
