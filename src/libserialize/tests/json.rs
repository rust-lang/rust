#[allow(unused_extern_crates)]
extern crate serialize as rustc_serialize;

use rustc_serialize::{Encodable, Decodable};
use rustc_serialize::json;
use json::Json::*;
use json::ErrorCode::*;
use json::ParserError::*;
use json::DecoderError::*;
use json::JsonEvent::*;
use json::{Json, from_str, DecodeResult, DecoderError, JsonEvent, Parser, StackElement,
           Decoder, Encoder, EncoderError};

use Animal::*;
use std::{i64, u64, f32, f64};
use std::io::prelude::*;
use std::collections::BTreeMap;
use std::string;

#[derive(RustcDecodable, Eq, PartialEq, Debug)]
struct OptionData {
    opt: Option<usize>,
}

#[test]
fn test_decode_option_none() {
    let s ="{}";
    let obj: OptionData = json::decode(s).unwrap();
    assert_eq!(obj, OptionData { opt: None });
}

#[test]
fn test_decode_option_some() {
    let s = "{ \"opt\": 10 }";
    let obj: OptionData = json::decode(s).unwrap();
    assert_eq!(obj, OptionData { opt: Some(10) });
}

#[test]
fn test_decode_option_malformed() {
    check_err::<OptionData>("{ \"opt\": [] }",
                            ExpectedError("Number".to_string(), "[]".to_string()));
    check_err::<OptionData>("{ \"opt\": false }",
                            ExpectedError("Number".to_string(), "false".to_string()));
}

#[derive(PartialEq, RustcEncodable, RustcDecodable, Debug)]
enum Animal {
    Dog,
    Frog(string::String, isize)
}

#[derive(PartialEq, RustcEncodable, RustcDecodable, Debug)]
struct Inner {
    a: (),
    b: usize,
    c: Vec<string::String>,
}

#[derive(PartialEq, RustcEncodable, RustcDecodable, Debug)]
struct Outer {
    inner: Vec<Inner>,
}

fn mk_object(items: &[(string::String, Json)]) -> Json {
    let mut d = BTreeMap::new();

    for item in items {
        match *item {
            (ref key, ref value) => { d.insert((*key).clone(), (*value).clone()); },
        }
    };

    Object(d)
}

#[test]
fn test_from_str_trait() {
    let s = "null";
    assert!(s.parse::<Json>().unwrap() == s.parse().unwrap());
}

#[test]
fn test_write_null() {
    assert_eq!(Null.to_string(), "null");
    assert_eq!(Null.pretty().to_string(), "null");
}

#[test]
fn test_write_i64() {
    assert_eq!(U64(0).to_string(), "0");
    assert_eq!(U64(0).pretty().to_string(), "0");

    assert_eq!(U64(1234).to_string(), "1234");
    assert_eq!(U64(1234).pretty().to_string(), "1234");

    assert_eq!(I64(-5678).to_string(), "-5678");
    assert_eq!(I64(-5678).pretty().to_string(), "-5678");

    assert_eq!(U64(7650007200025252000).to_string(), "7650007200025252000");
    assert_eq!(U64(7650007200025252000).pretty().to_string(), "7650007200025252000");
}

#[test]
fn test_write_f64() {
    assert_eq!(F64(3.0).to_string(), "3.0");
    assert_eq!(F64(3.0).pretty().to_string(), "3.0");

    assert_eq!(F64(3.1).to_string(), "3.1");
    assert_eq!(F64(3.1).pretty().to_string(), "3.1");

    assert_eq!(F64(-1.5).to_string(), "-1.5");
    assert_eq!(F64(-1.5).pretty().to_string(), "-1.5");

    assert_eq!(F64(0.5).to_string(), "0.5");
    assert_eq!(F64(0.5).pretty().to_string(), "0.5");

    assert_eq!(F64(f64::NAN).to_string(), "null");
    assert_eq!(F64(f64::NAN).pretty().to_string(), "null");

    assert_eq!(F64(f64::INFINITY).to_string(), "null");
    assert_eq!(F64(f64::INFINITY).pretty().to_string(), "null");

    assert_eq!(F64(f64::NEG_INFINITY).to_string(), "null");
    assert_eq!(F64(f64::NEG_INFINITY).pretty().to_string(), "null");
}

#[test]
fn test_write_str() {
    assert_eq!(String("".to_string()).to_string(), "\"\"");
    assert_eq!(String("".to_string()).pretty().to_string(), "\"\"");

    assert_eq!(String("homura".to_string()).to_string(), "\"homura\"");
    assert_eq!(String("madoka".to_string()).pretty().to_string(), "\"madoka\"");
}

#[test]
fn test_write_bool() {
    assert_eq!(Boolean(true).to_string(), "true");
    assert_eq!(Boolean(true).pretty().to_string(), "true");

    assert_eq!(Boolean(false).to_string(), "false");
    assert_eq!(Boolean(false).pretty().to_string(), "false");
}

#[test]
fn test_write_array() {
    assert_eq!(Array(vec![]).to_string(), "[]");
    assert_eq!(Array(vec![]).pretty().to_string(), "[]");

    assert_eq!(Array(vec![Boolean(true)]).to_string(), "[true]");
    assert_eq!(
        Array(vec![Boolean(true)]).pretty().to_string(),
        "\
        [\n  \
            true\n\
        ]"
    );

    let long_test_array = Array(vec![
        Boolean(false),
        Null,
        Array(vec![String("foo\nbar".to_string()), F64(3.5)])]);

    assert_eq!(long_test_array.to_string(),
        "[false,null,[\"foo\\nbar\",3.5]]");
    assert_eq!(
        long_test_array.pretty().to_string(),
        "\
        [\n  \
            false,\n  \
            null,\n  \
            [\n    \
                \"foo\\nbar\",\n    \
                3.5\n  \
            ]\n\
        ]"
    );
}

#[test]
fn test_write_object() {
    assert_eq!(mk_object(&[]).to_string(), "{}");
    assert_eq!(mk_object(&[]).pretty().to_string(), "{}");

    assert_eq!(
        mk_object(&[
            ("a".to_string(), Boolean(true))
        ]).to_string(),
        "{\"a\":true}"
    );
    assert_eq!(
        mk_object(&[("a".to_string(), Boolean(true))]).pretty().to_string(),
        "\
        {\n  \
            \"a\": true\n\
        }"
    );

    let complex_obj = mk_object(&[
            ("b".to_string(), Array(vec![
                mk_object(&[("c".to_string(), String("\x0c\r".to_string()))]),
                mk_object(&[("d".to_string(), String("".to_string()))])
            ]))
        ]);

    assert_eq!(
        complex_obj.to_string(),
        "{\
            \"b\":[\
                {\"c\":\"\\f\\r\"},\
                {\"d\":\"\"}\
            ]\
        }"
    );
    assert_eq!(
        complex_obj.pretty().to_string(),
        "\
        {\n  \
            \"b\": [\n    \
                {\n      \
                    \"c\": \"\\f\\r\"\n    \
                },\n    \
                {\n      \
                    \"d\": \"\"\n    \
                }\n  \
            ]\n\
        }"
    );

    let a = mk_object(&[
        ("a".to_string(), Boolean(true)),
        ("b".to_string(), Array(vec![
            mk_object(&[("c".to_string(), String("\x0c\r".to_string()))]),
            mk_object(&[("d".to_string(), String("".to_string()))])
        ]))
    ]);

    // We can't compare the strings directly because the object fields be
    // printed in a different order.
    assert_eq!(a.clone(), a.to_string().parse().unwrap());
    assert_eq!(a.clone(), a.pretty().to_string().parse().unwrap());
}

#[test]
fn test_write_enum() {
    let animal = Dog;
    assert_eq!(
        json::as_json(&animal).to_string(),
        "\"Dog\""
    );
    assert_eq!(
        json::as_pretty_json(&animal).to_string(),
        "\"Dog\""
    );

    let animal = Frog("Henry".to_string(), 349);
    assert_eq!(
        json::as_json(&animal).to_string(),
        "{\"variant\":\"Frog\",\"fields\":[\"Henry\",349]}"
    );
    assert_eq!(
        json::as_pretty_json(&animal).to_string(),
        "{\n  \
           \"variant\": \"Frog\",\n  \
           \"fields\": [\n    \
             \"Henry\",\n    \
             349\n  \
           ]\n\
         }"
    );
}

macro_rules! check_encoder_for_simple {
    ($value:expr, $expected:expr) => ({
        let s = json::as_json(&$value).to_string();
        assert_eq!(s, $expected);

        let s = json::as_pretty_json(&$value).to_string();
        assert_eq!(s, $expected);
    })
}

#[test]
fn test_write_some() {
    check_encoder_for_simple!(Some("jodhpurs".to_string()), "\"jodhpurs\"");
}

#[test]
fn test_write_none() {
    check_encoder_for_simple!(None::<string::String>, "null");
}

#[test]
fn test_write_char() {
    check_encoder_for_simple!('a', "\"a\"");
    check_encoder_for_simple!('\t', "\"\\t\"");
    check_encoder_for_simple!('\u{0000}', "\"\\u0000\"");
    check_encoder_for_simple!('\u{001b}', "\"\\u001b\"");
    check_encoder_for_simple!('\u{007f}', "\"\\u007f\"");
    check_encoder_for_simple!('\u{00a0}', "\"\u{00a0}\"");
    check_encoder_for_simple!('\u{abcd}', "\"\u{abcd}\"");
    check_encoder_for_simple!('\u{10ffff}', "\"\u{10ffff}\"");
}

#[test]
fn test_trailing_characters() {
    assert_eq!(from_str("nulla"),  Err(SyntaxError(TrailingCharacters, 1, 5)));
    assert_eq!(from_str("truea"),  Err(SyntaxError(TrailingCharacters, 1, 5)));
    assert_eq!(from_str("falsea"), Err(SyntaxError(TrailingCharacters, 1, 6)));
    assert_eq!(from_str("1a"),     Err(SyntaxError(TrailingCharacters, 1, 2)));
    assert_eq!(from_str("[]a"),    Err(SyntaxError(TrailingCharacters, 1, 3)));
    assert_eq!(from_str("{}a"),    Err(SyntaxError(TrailingCharacters, 1, 3)));
}

#[test]
fn test_read_identifiers() {
    assert_eq!(from_str("n"),    Err(SyntaxError(InvalidSyntax, 1, 2)));
    assert_eq!(from_str("nul"),  Err(SyntaxError(InvalidSyntax, 1, 4)));
    assert_eq!(from_str("t"),    Err(SyntaxError(InvalidSyntax, 1, 2)));
    assert_eq!(from_str("truz"), Err(SyntaxError(InvalidSyntax, 1, 4)));
    assert_eq!(from_str("f"),    Err(SyntaxError(InvalidSyntax, 1, 2)));
    assert_eq!(from_str("faz"),  Err(SyntaxError(InvalidSyntax, 1, 3)));

    assert_eq!(from_str("null"), Ok(Null));
    assert_eq!(from_str("true"), Ok(Boolean(true)));
    assert_eq!(from_str("false"), Ok(Boolean(false)));
    assert_eq!(from_str(" null "), Ok(Null));
    assert_eq!(from_str(" true "), Ok(Boolean(true)));
    assert_eq!(from_str(" false "), Ok(Boolean(false)));
}

#[test]
fn test_decode_identifiers() {
    let v: () = json::decode("null").unwrap();
    assert_eq!(v, ());

    let v: bool = json::decode("true").unwrap();
    assert_eq!(v, true);

    let v: bool = json::decode("false").unwrap();
    assert_eq!(v, false);
}

#[test]
fn test_read_number() {
    assert_eq!(from_str("+"),   Err(SyntaxError(InvalidSyntax, 1, 1)));
    assert_eq!(from_str("."),   Err(SyntaxError(InvalidSyntax, 1, 1)));
    assert_eq!(from_str("NaN"), Err(SyntaxError(InvalidSyntax, 1, 1)));
    assert_eq!(from_str("-"),   Err(SyntaxError(InvalidNumber, 1, 2)));
    assert_eq!(from_str("00"),  Err(SyntaxError(InvalidNumber, 1, 2)));
    assert_eq!(from_str("1."),  Err(SyntaxError(InvalidNumber, 1, 3)));
    assert_eq!(from_str("1e"),  Err(SyntaxError(InvalidNumber, 1, 3)));
    assert_eq!(from_str("1e+"), Err(SyntaxError(InvalidNumber, 1, 4)));

    assert_eq!(from_str("18446744073709551616"), Err(SyntaxError(InvalidNumber, 1, 20)));
    assert_eq!(from_str("-9223372036854775809"), Err(SyntaxError(InvalidNumber, 1, 21)));

    assert_eq!(from_str("3"), Ok(U64(3)));
    assert_eq!(from_str("3.1"), Ok(F64(3.1)));
    assert_eq!(from_str("-1.2"), Ok(F64(-1.2)));
    assert_eq!(from_str("0.4"), Ok(F64(0.4)));
    assert_eq!(from_str("0.4e5"), Ok(F64(0.4e5)));
    assert_eq!(from_str("0.4e+15"), Ok(F64(0.4e15)));
    assert_eq!(from_str("0.4e-01"), Ok(F64(0.4e-01)));
    assert_eq!(from_str(" 3 "), Ok(U64(3)));

    assert_eq!(from_str("-9223372036854775808"), Ok(I64(i64::MIN)));
    assert_eq!(from_str("9223372036854775807"), Ok(U64(i64::MAX as u64)));
    assert_eq!(from_str("18446744073709551615"), Ok(U64(u64::MAX)));
}

#[test]
fn test_decode_numbers() {
    let v: f64 = json::decode("3").unwrap();
    assert_eq!(v, 3.0);

    let v: f64 = json::decode("3.1").unwrap();
    assert_eq!(v, 3.1);

    let v: f64 = json::decode("-1.2").unwrap();
    assert_eq!(v, -1.2);

    let v: f64 = json::decode("0.4").unwrap();
    assert_eq!(v, 0.4);

    let v: f64 = json::decode("0.4e5").unwrap();
    assert_eq!(v, 0.4e5);

    let v: f64 = json::decode("0.4e15").unwrap();
    assert_eq!(v, 0.4e15);

    let v: f64 = json::decode("0.4e-01").unwrap();
    assert_eq!(v, 0.4e-01);

    let v: u64 = json::decode("0").unwrap();
    assert_eq!(v, 0);

    let v: u64 = json::decode("18446744073709551615").unwrap();
    assert_eq!(v, u64::MAX);

    let v: i64 = json::decode("-9223372036854775808").unwrap();
    assert_eq!(v, i64::MIN);

    let v: i64 = json::decode("9223372036854775807").unwrap();
    assert_eq!(v, i64::MAX);

    let res: DecodeResult<i64> = json::decode("765.25");
    assert_eq!(res, Err(ExpectedError("Integer".to_string(),
                                      "765.25".to_string())));
}

#[test]
fn test_read_str() {
    assert_eq!(from_str("\""),    Err(SyntaxError(EOFWhileParsingString, 1, 2)));
    assert_eq!(from_str("\"lol"), Err(SyntaxError(EOFWhileParsingString, 1, 5)));

    assert_eq!(from_str("\"\""), Ok(String("".to_string())));
    assert_eq!(from_str("\"foo\""), Ok(String("foo".to_string())));
    assert_eq!(from_str("\"\\\"\""), Ok(String("\"".to_string())));
    assert_eq!(from_str("\"\\b\""), Ok(String("\x08".to_string())));
    assert_eq!(from_str("\"\\n\""), Ok(String("\n".to_string())));
    assert_eq!(from_str("\"\\r\""), Ok(String("\r".to_string())));
    assert_eq!(from_str("\"\\t\""), Ok(String("\t".to_string())));
    assert_eq!(from_str(" \"foo\" "), Ok(String("foo".to_string())));
    assert_eq!(from_str("\"\\u12ab\""), Ok(String("\u{12ab}".to_string())));
    assert_eq!(from_str("\"\\uAB12\""), Ok(String("\u{AB12}".to_string())));
}

#[test]
fn test_decode_str() {
    let s = [("\"\"", ""),
             ("\"foo\"", "foo"),
             ("\"\\\"\"", "\""),
             ("\"\\b\"", "\x08"),
             ("\"\\n\"", "\n"),
             ("\"\\r\"", "\r"),
             ("\"\\t\"", "\t"),
             ("\"\\u12ab\"", "\u{12ab}"),
             ("\"\\uAB12\"", "\u{AB12}")];

    for &(i, o) in &s {
        let v: string::String = json::decode(i).unwrap();
        assert_eq!(v, o);
    }
}

#[test]
fn test_read_array() {
    assert_eq!(from_str("["),     Err(SyntaxError(EOFWhileParsingValue, 1, 2)));
    assert_eq!(from_str("[1"),    Err(SyntaxError(EOFWhileParsingArray, 1, 3)));
    assert_eq!(from_str("[1,"),   Err(SyntaxError(EOFWhileParsingValue, 1, 4)));
    assert_eq!(from_str("[1,]"),  Err(SyntaxError(InvalidSyntax,        1, 4)));
    assert_eq!(from_str("[6 7]"), Err(SyntaxError(InvalidSyntax,        1, 4)));

    assert_eq!(from_str("[]"), Ok(Array(vec![])));
    assert_eq!(from_str("[ ]"), Ok(Array(vec![])));
    assert_eq!(from_str("[true]"), Ok(Array(vec![Boolean(true)])));
    assert_eq!(from_str("[ false ]"), Ok(Array(vec![Boolean(false)])));
    assert_eq!(from_str("[null]"), Ok(Array(vec![Null])));
    assert_eq!(from_str("[3, 1]"),
                 Ok(Array(vec![U64(3), U64(1)])));
    assert_eq!(from_str("\n[3, 2]\n"),
                 Ok(Array(vec![U64(3), U64(2)])));
    assert_eq!(from_str("[2, [4, 1]]"),
           Ok(Array(vec![U64(2), Array(vec![U64(4), U64(1)])])));
}

#[test]
fn test_decode_array() {
    let v: Vec<()> = json::decode("[]").unwrap();
    assert_eq!(v, []);

    let v: Vec<()> = json::decode("[null]").unwrap();
    assert_eq!(v, [()]);

    let v: Vec<bool> = json::decode("[true]").unwrap();
    assert_eq!(v, [true]);

    let v: Vec<isize> = json::decode("[3, 1]").unwrap();
    assert_eq!(v, [3, 1]);

    let v: Vec<Vec<usize>> = json::decode("[[3], [1, 2]]").unwrap();
    assert_eq!(v, [vec![3], vec![1, 2]]);
}

#[test]
fn test_decode_tuple() {
    let t: (usize, usize, usize) = json::decode("[1, 2, 3]").unwrap();
    assert_eq!(t, (1, 2, 3));

    let t: (usize, string::String) = json::decode("[1, \"two\"]").unwrap();
    assert_eq!(t, (1, "two".to_string()));
}

#[test]
fn test_decode_tuple_malformed_types() {
    assert!(json::decode::<(usize, string::String)>("[1, 2]").is_err());
}

#[test]
fn test_decode_tuple_malformed_length() {
    assert!(json::decode::<(usize, usize)>("[1, 2, 3]").is_err());
}

#[test]
fn test_read_object() {
    assert_eq!(from_str("{"),       Err(SyntaxError(EOFWhileParsingObject, 1, 2)));
    assert_eq!(from_str("{ "),      Err(SyntaxError(EOFWhileParsingObject, 1, 3)));
    assert_eq!(from_str("{1"),      Err(SyntaxError(KeyMustBeAString,      1, 2)));
    assert_eq!(from_str("{ \"a\""), Err(SyntaxError(EOFWhileParsingObject, 1, 6)));
    assert_eq!(from_str("{\"a\""),  Err(SyntaxError(EOFWhileParsingObject, 1, 5)));
    assert_eq!(from_str("{\"a\" "), Err(SyntaxError(EOFWhileParsingObject, 1, 6)));

    assert_eq!(from_str("{\"a\" 1"),   Err(SyntaxError(ExpectedColon,         1, 6)));
    assert_eq!(from_str("{\"a\":"),    Err(SyntaxError(EOFWhileParsingValue,  1, 6)));
    assert_eq!(from_str("{\"a\":1"),   Err(SyntaxError(EOFWhileParsingObject, 1, 7)));
    assert_eq!(from_str("{\"a\":1 1"), Err(SyntaxError(InvalidSyntax,         1, 8)));
    assert_eq!(from_str("{\"a\":1,"),  Err(SyntaxError(EOFWhileParsingObject, 1, 8)));

    assert_eq!(from_str("{}").unwrap(), mk_object(&[]));
    assert_eq!(from_str("{\"a\": 3}").unwrap(),
                mk_object(&[("a".to_string(), U64(3))]));

    assert_eq!(from_str(
                    "{ \"a\": null, \"b\" : true }").unwrap(),
                mk_object(&[
                    ("a".to_string(), Null),
                    ("b".to_string(), Boolean(true))]));
    assert_eq!(from_str("\n{ \"a\": null, \"b\" : true }\n").unwrap(),
                mk_object(&[
                    ("a".to_string(), Null),
                    ("b".to_string(), Boolean(true))]));
    assert_eq!(from_str(
                    "{\"a\" : 1.0 ,\"b\": [ true ]}").unwrap(),
                mk_object(&[
                    ("a".to_string(), F64(1.0)),
                    ("b".to_string(), Array(vec![Boolean(true)]))
                ]));
    assert_eq!(from_str(
                    "{\
                        \"a\": 1.0, \
                        \"b\": [\
                            true,\
                            \"foo\\nbar\", \
                            { \"c\": {\"d\": null} } \
                        ]\
                    }").unwrap(),
                mk_object(&[
                    ("a".to_string(), F64(1.0)),
                    ("b".to_string(), Array(vec![
                        Boolean(true),
                        String("foo\nbar".to_string()),
                        mk_object(&[
                            ("c".to_string(), mk_object(&[("d".to_string(), Null)]))
                        ])
                    ]))
                ]));
}

#[test]
fn test_decode_struct() {
    let s = "{
        \"inner\": [
            { \"a\": null, \"b\": 2, \"c\": [\"abc\", \"xyz\"] }
        ]
    }";

    let v: Outer = json::decode(s).unwrap();
    assert_eq!(
        v,
        Outer {
            inner: vec![
                Inner { a: (), b: 2, c: vec!["abc".to_string(), "xyz".to_string()] }
            ]
        }
    );
}

#[derive(RustcDecodable)]
struct FloatStruct {
    f: f64,
    a: Vec<f64>
}
#[test]
fn test_decode_struct_with_nan() {
    let s = "{\"f\":null,\"a\":[null,123]}";
    let obj: FloatStruct = json::decode(s).unwrap();
    assert!(obj.f.is_nan());
    assert!(obj.a[0].is_nan());
    assert_eq!(obj.a[1], 123f64);
}

#[test]
fn test_decode_option() {
    let value: Option<string::String> = json::decode("null").unwrap();
    assert_eq!(value, None);

    let value: Option<string::String> = json::decode("\"jodhpurs\"").unwrap();
    assert_eq!(value, Some("jodhpurs".to_string()));
}

#[test]
fn test_decode_enum() {
    let value: Animal = json::decode("\"Dog\"").unwrap();
    assert_eq!(value, Dog);

    let s = "{\"variant\":\"Frog\",\"fields\":[\"Henry\",349]}";
    let value: Animal = json::decode(s).unwrap();
    assert_eq!(value, Frog("Henry".to_string(), 349));
}

#[test]
fn test_decode_map() {
    let s = "{\"a\": \"Dog\", \"b\": {\"variant\":\"Frog\",\
              \"fields\":[\"Henry\", 349]}}";
    let mut map: BTreeMap<string::String, Animal> = json::decode(s).unwrap();

    assert_eq!(map.remove(&"a".to_string()), Some(Dog));
    assert_eq!(map.remove(&"b".to_string()), Some(Frog("Henry".to_string(), 349)));
}

#[test]
fn test_multiline_errors() {
    assert_eq!(from_str("{\n  \"foo\":\n \"bar\""),
        Err(SyntaxError(EOFWhileParsingObject, 3, 8)));
}

#[derive(RustcDecodable)]
#[allow(dead_code)]
struct DecodeStruct {
    x: f64,
    y: bool,
    z: string::String,
    w: Vec<DecodeStruct>
}
#[derive(RustcDecodable)]
enum DecodeEnum {
    A(f64),
    B(string::String)
}
fn check_err<T: Decodable>(to_parse: &'static str, expected: DecoderError) {
    let res: DecodeResult<T> = match from_str(to_parse) {
        Err(e) => Err(ParseError(e)),
        Ok(json) => Decodable::decode(&mut Decoder::new(json))
    };
    match res {
        Ok(_) => panic!("`{:?}` parsed & decoded ok, expecting error `{:?}`",
                           to_parse, expected),
        Err(ParseError(e)) => panic!("`{:?}` is not valid json: {:?}",
                                        to_parse, e),
        Err(e) => {
            assert_eq!(e, expected);
        }
    }
}
#[test]
fn test_decode_errors_struct() {
    check_err::<DecodeStruct>("[]", ExpectedError("Object".to_string(), "[]".to_string()));
    check_err::<DecodeStruct>("{\"x\": true, \"y\": true, \"z\": \"\", \"w\": []}",
                              ExpectedError("Number".to_string(), "true".to_string()));
    check_err::<DecodeStruct>("{\"x\": 1, \"y\": [], \"z\": \"\", \"w\": []}",
                              ExpectedError("Boolean".to_string(), "[]".to_string()));
    check_err::<DecodeStruct>("{\"x\": 1, \"y\": true, \"z\": {}, \"w\": []}",
                              ExpectedError("String".to_string(), "{}".to_string()));
    check_err::<DecodeStruct>("{\"x\": 1, \"y\": true, \"z\": \"\", \"w\": null}",
                              ExpectedError("Array".to_string(), "null".to_string()));
    check_err::<DecodeStruct>("{\"x\": 1, \"y\": true, \"z\": \"\"}",
                              MissingFieldError("w".to_string()));
}
#[test]
fn test_decode_errors_enum() {
    check_err::<DecodeEnum>("{}",
                            MissingFieldError("variant".to_string()));
    check_err::<DecodeEnum>("{\"variant\": 1}",
                            ExpectedError("String".to_string(), "1".to_string()));
    check_err::<DecodeEnum>("{\"variant\": \"A\"}",
                            MissingFieldError("fields".to_string()));
    check_err::<DecodeEnum>("{\"variant\": \"A\", \"fields\": null}",
                            ExpectedError("Array".to_string(), "null".to_string()));
    check_err::<DecodeEnum>("{\"variant\": \"C\", \"fields\": []}",
                            UnknownVariantError("C".to_string()));
}

#[test]
fn test_find(){
    let json_value = from_str("{\"dog\" : \"cat\"}").unwrap();
    let found_str = json_value.find("dog");
    assert!(found_str.unwrap().as_string().unwrap() == "cat");
}

#[test]
fn test_find_path(){
    let json_value = from_str("{\"dog\":{\"cat\": {\"mouse\" : \"cheese\"}}}").unwrap();
    let found_str = json_value.find_path(&["dog", "cat", "mouse"]);
    assert!(found_str.unwrap().as_string().unwrap() == "cheese");
}

#[test]
fn test_search(){
    let json_value = from_str("{\"dog\":{\"cat\": {\"mouse\" : \"cheese\"}}}").unwrap();
    let found_str = json_value.search("mouse").and_then(|j| j.as_string());
    assert!(found_str.unwrap() == "cheese");
}

#[test]
fn test_index(){
    let json_value = from_str("{\"animals\":[\"dog\",\"cat\",\"mouse\"]}").unwrap();
    let ref array = json_value["animals"];
    assert_eq!(array[0].as_string().unwrap(), "dog");
    assert_eq!(array[1].as_string().unwrap(), "cat");
    assert_eq!(array[2].as_string().unwrap(), "mouse");
}

#[test]
fn test_is_object(){
    let json_value = from_str("{}").unwrap();
    assert!(json_value.is_object());
}

#[test]
fn test_as_object(){
    let json_value = from_str("{}").unwrap();
    let json_object = json_value.as_object();
    assert!(json_object.is_some());
}

#[test]
fn test_is_array(){
    let json_value = from_str("[1, 2, 3]").unwrap();
    assert!(json_value.is_array());
}

#[test]
fn test_as_array(){
    let json_value = from_str("[1, 2, 3]").unwrap();
    let json_array = json_value.as_array();
    let expected_length = 3;
    assert!(json_array.is_some() && json_array.unwrap().len() == expected_length);
}

#[test]
fn test_is_string(){
    let json_value = from_str("\"dog\"").unwrap();
    assert!(json_value.is_string());
}

#[test]
fn test_as_string(){
    let json_value = from_str("\"dog\"").unwrap();
    let json_str = json_value.as_string();
    let expected_str = "dog";
    assert_eq!(json_str, Some(expected_str));
}

#[test]
fn test_is_number(){
    let json_value = from_str("12").unwrap();
    assert!(json_value.is_number());
}

#[test]
fn test_is_i64(){
    let json_value = from_str("-12").unwrap();
    assert!(json_value.is_i64());

    let json_value = from_str("12").unwrap();
    assert!(!json_value.is_i64());

    let json_value = from_str("12.0").unwrap();
    assert!(!json_value.is_i64());
}

#[test]
fn test_is_u64(){
    let json_value = from_str("12").unwrap();
    assert!(json_value.is_u64());

    let json_value = from_str("-12").unwrap();
    assert!(!json_value.is_u64());

    let json_value = from_str("12.0").unwrap();
    assert!(!json_value.is_u64());
}

#[test]
fn test_is_f64(){
    let json_value = from_str("12").unwrap();
    assert!(!json_value.is_f64());

    let json_value = from_str("-12").unwrap();
    assert!(!json_value.is_f64());

    let json_value = from_str("12.0").unwrap();
    assert!(json_value.is_f64());

    let json_value = from_str("-12.0").unwrap();
    assert!(json_value.is_f64());
}

#[test]
fn test_as_i64(){
    let json_value = from_str("-12").unwrap();
    let json_num = json_value.as_i64();
    assert_eq!(json_num, Some(-12));
}

#[test]
fn test_as_u64(){
    let json_value = from_str("12").unwrap();
    let json_num = json_value.as_u64();
    assert_eq!(json_num, Some(12));
}

#[test]
fn test_as_f64(){
    let json_value = from_str("12.0").unwrap();
    let json_num = json_value.as_f64();
    assert_eq!(json_num, Some(12f64));
}

#[test]
fn test_is_boolean(){
    let json_value = from_str("false").unwrap();
    assert!(json_value.is_boolean());
}

#[test]
fn test_as_boolean(){
    let json_value = from_str("false").unwrap();
    let json_bool = json_value.as_boolean();
    let expected_bool = false;
    assert!(json_bool.is_some() && json_bool.unwrap() == expected_bool);
}

#[test]
fn test_is_null(){
    let json_value = from_str("null").unwrap();
    assert!(json_value.is_null());
}

#[test]
fn test_as_null(){
    let json_value = from_str("null").unwrap();
    let json_null = json_value.as_null();
    let expected_null = ();
    assert!(json_null.is_some() && json_null.unwrap() == expected_null);
}

#[test]
fn test_encode_hashmap_with_numeric_key() {
    use std::str::from_utf8;
    use std::collections::HashMap;
    let mut hm: HashMap<usize, bool> = HashMap::new();
    hm.insert(1, true);
    let mut mem_buf = Vec::new();
    write!(&mut mem_buf, "{}", json::as_pretty_json(&hm)).unwrap();
    let json_str = from_utf8(&mem_buf[..]).unwrap();
    match from_str(json_str) {
        Err(_) => panic!("Unable to parse json_str: {:?}", json_str),
        _ => {} // it parsed and we are good to go
    }
}

#[test]
fn test_prettyencode_hashmap_with_numeric_key() {
    use std::str::from_utf8;
    use std::collections::HashMap;
    let mut hm: HashMap<usize, bool> = HashMap::new();
    hm.insert(1, true);
    let mut mem_buf = Vec::new();
    write!(&mut mem_buf, "{}", json::as_pretty_json(&hm)).unwrap();
    let json_str = from_utf8(&mem_buf[..]).unwrap();
    match from_str(json_str) {
        Err(_) => panic!("Unable to parse json_str: {:?}", json_str),
        _ => {} // it parsed and we are good to go
    }
}

#[test]
fn test_prettyencoder_indent_level_param() {
    use std::str::from_utf8;
    use std::collections::BTreeMap;

    let mut tree = BTreeMap::new();

    tree.insert("hello".to_string(), String("guten tag".to_string()));
    tree.insert("goodbye".to_string(), String("sayonara".to_string()));

    let json = Array(
        // The following layout below should look a lot like
        // the pretty-printed JSON (indent * x)
        vec!
        ( // 0x
            String("greetings".to_string()), // 1x
            Object(tree), // 1x + 2x + 2x + 1x
        ) // 0x
        // End JSON array (7 lines)
    );

    // Helper function for counting indents
    fn indents(source: &str) -> usize {
        let trimmed = source.trim_start_matches(' ');
        source.len() - trimmed.len()
    }

    // Test up to 4 spaces of indents (more?)
    for i in 0..4 {
        let mut writer = Vec::new();
        write!(&mut writer, "{}",
                json::as_pretty_json(&json).indent(i)).unwrap();

        let printed = from_utf8(&writer[..]).unwrap();

        // Check for indents at each line
        let lines: Vec<&str> = printed.lines().collect();
        assert_eq!(lines.len(), 7); // JSON should be 7 lines

        assert_eq!(indents(lines[0]), 0 * i); // [
        assert_eq!(indents(lines[1]), 1 * i); //   "greetings",
        assert_eq!(indents(lines[2]), 1 * i); //   {
        assert_eq!(indents(lines[3]), 2 * i); //     "hello": "guten tag",
        assert_eq!(indents(lines[4]), 2 * i); //     "goodbye": "sayonara"
        assert_eq!(indents(lines[5]), 1 * i); //   },
        assert_eq!(indents(lines[6]), 0 * i); // ]

        // Finally, test that the pretty-printed JSON is valid
        from_str(printed).ok().expect("Pretty-printed JSON is invalid!");
    }
}

#[test]
fn test_hashmap_with_enum_key() {
    use std::collections::HashMap;
    #[derive(RustcEncodable, Eq, Hash, PartialEq, RustcDecodable, Debug)]
    enum Enum {
        Foo,
        #[allow(dead_code)]
        Bar,
    }
    let mut map = HashMap::new();
    map.insert(Enum::Foo, 0);
    let result = json::encode(&map).unwrap();
    assert_eq!(&result[..], r#"{"Foo":0}"#);
    let decoded: HashMap<Enum, _> = json::decode(&result).unwrap();
    assert_eq!(map, decoded);
}

#[test]
fn test_hashmap_with_numeric_key_can_handle_double_quote_delimited_key() {
    use std::collections::HashMap;
    let json_str = "{\"1\":true}";
    let json_obj = match from_str(json_str) {
        Err(_) => panic!("Unable to parse json_str: {:?}", json_str),
        Ok(o) => o
    };
    let mut decoder = Decoder::new(json_obj);
    let _hm: HashMap<usize, bool> = Decodable::decode(&mut decoder).unwrap();
}

#[test]
fn test_hashmap_with_numeric_key_will_error_with_string_keys() {
    use std::collections::HashMap;
    let json_str = "{\"a\":true}";
    let json_obj = match from_str(json_str) {
        Err(_) => panic!("Unable to parse json_str: {:?}", json_str),
        Ok(o) => o
    };
    let mut decoder = Decoder::new(json_obj);
    let result: Result<HashMap<usize, bool>, DecoderError> = Decodable::decode(&mut decoder);
    assert_eq!(result, Err(ExpectedError("Number".to_string(), "a".to_string())));
}

fn assert_stream_equal(src: &str,
                        expected: Vec<(JsonEvent, Vec<StackElement<'_>>)>) {
    let mut parser = Parser::new(src.chars());
    let mut i = 0;
    loop {
        let evt = match parser.next() {
            Some(e) => e,
            None => { break; }
        };
        let (ref expected_evt, ref expected_stack) = expected[i];
        if !parser.stack().is_equal_to(expected_stack) {
            panic!("Parser stack is not equal to {:?}", expected_stack);
        }
        assert_eq!(&evt, expected_evt);
        i+=1;
    }
}
#[test]
fn test_streaming_parser() {
    assert_stream_equal(
        r#"{ "foo":"bar", "array" : [0, 1, 2, 3, 4, 5], "idents":[null,true,false]}"#,
        vec![
            (ObjectStart,             vec![]),
              (StringValue("bar".to_string()),   vec![StackElement::Key("foo")]),
              (ArrayStart,            vec![StackElement::Key("array")]),
                (U64Value(0),         vec![StackElement::Key("array"), StackElement::Index(0)]),
                (U64Value(1),         vec![StackElement::Key("array"), StackElement::Index(1)]),
                (U64Value(2),         vec![StackElement::Key("array"), StackElement::Index(2)]),
                (U64Value(3),         vec![StackElement::Key("array"), StackElement::Index(3)]),
                (U64Value(4),         vec![StackElement::Key("array"), StackElement::Index(4)]),
                (U64Value(5),         vec![StackElement::Key("array"), StackElement::Index(5)]),
              (ArrayEnd,              vec![StackElement::Key("array")]),
              (ArrayStart,            vec![StackElement::Key("idents")]),
                (NullValue,           vec![StackElement::Key("idents"),
                                           StackElement::Index(0)]),
                (BooleanValue(true),  vec![StackElement::Key("idents"),
                                           StackElement::Index(1)]),
                (BooleanValue(false), vec![StackElement::Key("idents"),
                                           StackElement::Index(2)]),
              (ArrayEnd,              vec![StackElement::Key("idents")]),
            (ObjectEnd,               vec![]),
        ]
    );
}
fn last_event(src: &str) -> JsonEvent {
    let mut parser = Parser::new(src.chars());
    let mut evt = NullValue;
    loop {
        evt = match parser.next() {
            Some(e) => e,
            None => return evt,
        }
    }
}

#[test]
fn test_read_object_streaming() {
    assert_eq!(last_event("{ "),      Error(SyntaxError(EOFWhileParsingObject, 1, 3)));
    assert_eq!(last_event("{1"),      Error(SyntaxError(KeyMustBeAString,      1, 2)));
    assert_eq!(last_event("{ \"a\""), Error(SyntaxError(EOFWhileParsingObject, 1, 6)));
    assert_eq!(last_event("{\"a\""),  Error(SyntaxError(EOFWhileParsingObject, 1, 5)));
    assert_eq!(last_event("{\"a\" "), Error(SyntaxError(EOFWhileParsingObject, 1, 6)));

    assert_eq!(last_event("{\"a\" 1"),   Error(SyntaxError(ExpectedColon,         1, 6)));
    assert_eq!(last_event("{\"a\":"),    Error(SyntaxError(EOFWhileParsingValue,  1, 6)));
    assert_eq!(last_event("{\"a\":1"),   Error(SyntaxError(EOFWhileParsingObject, 1, 7)));
    assert_eq!(last_event("{\"a\":1 1"), Error(SyntaxError(InvalidSyntax,         1, 8)));
    assert_eq!(last_event("{\"a\":1,"),  Error(SyntaxError(EOFWhileParsingObject, 1, 8)));
    assert_eq!(last_event("{\"a\":1,}"), Error(SyntaxError(TrailingComma, 1, 8)));

    assert_stream_equal(
        "{}",
        vec![(ObjectStart, vec![]), (ObjectEnd, vec![])]
    );
    assert_stream_equal(
        "{\"a\": 3}",
        vec![
            (ObjectStart,        vec![]),
              (U64Value(3),      vec![StackElement::Key("a")]),
            (ObjectEnd,          vec![]),
        ]
    );
    assert_stream_equal(
        "{ \"a\": null, \"b\" : true }",
        vec![
            (ObjectStart,           vec![]),
              (NullValue,           vec![StackElement::Key("a")]),
              (BooleanValue(true),  vec![StackElement::Key("b")]),
            (ObjectEnd,             vec![]),
        ]
    );
    assert_stream_equal(
        "{\"a\" : 1.0 ,\"b\": [ true ]}",
        vec![
            (ObjectStart,           vec![]),
              (F64Value(1.0),       vec![StackElement::Key("a")]),
              (ArrayStart,          vec![StackElement::Key("b")]),
                (BooleanValue(true),vec![StackElement::Key("b"), StackElement::Index(0)]),
              (ArrayEnd,            vec![StackElement::Key("b")]),
            (ObjectEnd,             vec![]),
        ]
    );
    assert_stream_equal(
        r#"{
            "a": 1.0,
            "b": [
                true,
                "foo\nbar",
                { "c": {"d": null} }
            ]
        }"#,
        vec![
            (ObjectStart,                   vec![]),
              (F64Value(1.0),               vec![StackElement::Key("a")]),
              (ArrayStart,                  vec![StackElement::Key("b")]),
                (BooleanValue(true),        vec![StackElement::Key("b"),
                                                StackElement::Index(0)]),
                (StringValue("foo\nbar".to_string()),  vec![StackElement::Key("b"),
                                                            StackElement::Index(1)]),
                (ObjectStart,               vec![StackElement::Key("b"),
                                                 StackElement::Index(2)]),
                  (ObjectStart,             vec![StackElement::Key("b"),
                                                 StackElement::Index(2),
                                                 StackElement::Key("c")]),
                    (NullValue,             vec![StackElement::Key("b"),
                                                 StackElement::Index(2),
                                                 StackElement::Key("c"),
                                                 StackElement::Key("d")]),
                  (ObjectEnd,               vec![StackElement::Key("b"),
                                                 StackElement::Index(2),
                                                 StackElement::Key("c")]),
                (ObjectEnd,                 vec![StackElement::Key("b"),
                                                 StackElement::Index(2)]),
              (ArrayEnd,                    vec![StackElement::Key("b")]),
            (ObjectEnd,                     vec![]),
        ]
    );
}
#[test]
fn test_read_array_streaming() {
    assert_stream_equal(
        "[]",
        vec![
            (ArrayStart, vec![]),
            (ArrayEnd,   vec![]),
        ]
    );
    assert_stream_equal(
        "[ ]",
        vec![
            (ArrayStart, vec![]),
            (ArrayEnd,   vec![]),
        ]
    );
    assert_stream_equal(
        "[true]",
        vec![
            (ArrayStart,             vec![]),
                (BooleanValue(true), vec![StackElement::Index(0)]),
            (ArrayEnd,               vec![]),
        ]
    );
    assert_stream_equal(
        "[ false ]",
        vec![
            (ArrayStart,              vec![]),
                (BooleanValue(false), vec![StackElement::Index(0)]),
            (ArrayEnd,                vec![]),
        ]
    );
    assert_stream_equal(
        "[null]",
        vec![
            (ArrayStart,    vec![]),
                (NullValue, vec![StackElement::Index(0)]),
            (ArrayEnd,      vec![]),
        ]
    );
    assert_stream_equal(
        "[3, 1]",
        vec![
            (ArrayStart,      vec![]),
                (U64Value(3), vec![StackElement::Index(0)]),
                (U64Value(1), vec![StackElement::Index(1)]),
            (ArrayEnd,        vec![]),
        ]
    );
    assert_stream_equal(
        "\n[3, 2]\n",
        vec![
            (ArrayStart,      vec![]),
                (U64Value(3), vec![StackElement::Index(0)]),
                (U64Value(2), vec![StackElement::Index(1)]),
            (ArrayEnd,        vec![]),
        ]
    );
    assert_stream_equal(
        "[2, [4, 1]]",
        vec![
            (ArrayStart,           vec![]),
                (U64Value(2),      vec![StackElement::Index(0)]),
                (ArrayStart,       vec![StackElement::Index(1)]),
                    (U64Value(4),  vec![StackElement::Index(1), StackElement::Index(0)]),
                    (U64Value(1),  vec![StackElement::Index(1), StackElement::Index(1)]),
                (ArrayEnd,         vec![StackElement::Index(1)]),
            (ArrayEnd,             vec![]),
        ]
    );

    assert_eq!(last_event("["), Error(SyntaxError(EOFWhileParsingValue, 1,  2)));

    assert_eq!(from_str("["),     Err(SyntaxError(EOFWhileParsingValue, 1, 2)));
    assert_eq!(from_str("[1"),    Err(SyntaxError(EOFWhileParsingArray, 1, 3)));
    assert_eq!(from_str("[1,"),   Err(SyntaxError(EOFWhileParsingValue, 1, 4)));
    assert_eq!(from_str("[1,]"),  Err(SyntaxError(InvalidSyntax,        1, 4)));
    assert_eq!(from_str("[6 7]"), Err(SyntaxError(InvalidSyntax,        1, 4)));

}
#[test]
fn test_trailing_characters_streaming() {
    assert_eq!(last_event("nulla"),  Error(SyntaxError(TrailingCharacters, 1, 5)));
    assert_eq!(last_event("truea"),  Error(SyntaxError(TrailingCharacters, 1, 5)));
    assert_eq!(last_event("falsea"), Error(SyntaxError(TrailingCharacters, 1, 6)));
    assert_eq!(last_event("1a"),     Error(SyntaxError(TrailingCharacters, 1, 2)));
    assert_eq!(last_event("[]a"),    Error(SyntaxError(TrailingCharacters, 1, 3)));
    assert_eq!(last_event("{}a"),    Error(SyntaxError(TrailingCharacters, 1, 3)));
}
#[test]
fn test_read_identifiers_streaming() {
    assert_eq!(Parser::new("null".chars()).next(), Some(NullValue));
    assert_eq!(Parser::new("true".chars()).next(), Some(BooleanValue(true)));
    assert_eq!(Parser::new("false".chars()).next(), Some(BooleanValue(false)));

    assert_eq!(last_event("n"),    Error(SyntaxError(InvalidSyntax, 1, 2)));
    assert_eq!(last_event("nul"),  Error(SyntaxError(InvalidSyntax, 1, 4)));
    assert_eq!(last_event("t"),    Error(SyntaxError(InvalidSyntax, 1, 2)));
    assert_eq!(last_event("truz"), Error(SyntaxError(InvalidSyntax, 1, 4)));
    assert_eq!(last_event("f"),    Error(SyntaxError(InvalidSyntax, 1, 2)));
    assert_eq!(last_event("faz"),  Error(SyntaxError(InvalidSyntax, 1, 3)));
}

#[test]
fn test_to_json() {
    use std::collections::{HashMap,BTreeMap};
    use json::ToJson;

    let array2 = Array(vec![U64(1), U64(2)]);
    let array3 = Array(vec![U64(1), U64(2), U64(3)]);
    let object = {
        let mut tree_map = BTreeMap::new();
        tree_map.insert("a".to_string(), U64(1));
        tree_map.insert("b".to_string(), U64(2));
        Object(tree_map)
    };

    assert_eq!(array2.to_json(), array2);
    assert_eq!(object.to_json(), object);
    assert_eq!(3_isize.to_json(), I64(3));
    assert_eq!(4_i8.to_json(), I64(4));
    assert_eq!(5_i16.to_json(), I64(5));
    assert_eq!(6_i32.to_json(), I64(6));
    assert_eq!(7_i64.to_json(), I64(7));
    assert_eq!(8_usize.to_json(), U64(8));
    assert_eq!(9_u8.to_json(), U64(9));
    assert_eq!(10_u16.to_json(), U64(10));
    assert_eq!(11_u32.to_json(), U64(11));
    assert_eq!(12_u64.to_json(), U64(12));
    assert_eq!(13.0_f32.to_json(), F64(13.0_f64));
    assert_eq!(14.0_f64.to_json(), F64(14.0_f64));
    assert_eq!(().to_json(), Null);
    assert_eq!(f32::INFINITY.to_json(), Null);
    assert_eq!(f64::NAN.to_json(), Null);
    assert_eq!(true.to_json(), Boolean(true));
    assert_eq!(false.to_json(), Boolean(false));
    assert_eq!("abc".to_json(), String("abc".to_string()));
    assert_eq!("abc".to_string().to_json(), String("abc".to_string()));
    assert_eq!((1_usize, 2_usize).to_json(), array2);
    assert_eq!((1_usize, 2_usize, 3_usize).to_json(), array3);
    assert_eq!([1_usize, 2_usize].to_json(), array2);
    assert_eq!((&[1_usize, 2_usize, 3_usize]).to_json(), array3);
    assert_eq!((vec![1_usize, 2_usize]).to_json(), array2);
    assert_eq!(vec![1_usize, 2_usize, 3_usize].to_json(), array3);
    let mut tree_map = BTreeMap::new();
    tree_map.insert("a".to_string(), 1 as usize);
    tree_map.insert("b".to_string(), 2);
    assert_eq!(tree_map.to_json(), object);
    let mut hash_map = HashMap::new();
    hash_map.insert("a".to_string(), 1 as usize);
    hash_map.insert("b".to_string(), 2);
    assert_eq!(hash_map.to_json(), object);
    assert_eq!(Some(15).to_json(), I64(15));
    assert_eq!(Some(15 as usize).to_json(), U64(15));
    assert_eq!(None::<isize>.to_json(), Null);
}

#[test]
fn test_encode_hashmap_with_arbitrary_key() {
    use std::collections::HashMap;
    #[derive(PartialEq, Eq, Hash, RustcEncodable)]
    struct ArbitraryType(usize);
    let mut hm: HashMap<ArbitraryType, bool> = HashMap::new();
    hm.insert(ArbitraryType(1), true);
    let mut mem_buf = string::String::new();
    let mut encoder = Encoder::new(&mut mem_buf);
    let result = hm.encode(&mut encoder);
    match result.unwrap_err() {
        EncoderError::BadHashmapKey => (),
        _ => panic!("expected bad hash map key")
    }
}
