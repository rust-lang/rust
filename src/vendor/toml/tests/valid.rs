extern crate rustc_serialize;
extern crate toml;

use std::collections::BTreeMap;
use rustc_serialize::json::Json;

use toml::{Parser, Value};
use toml::Value::{Table, Integer, Float, Boolean, Datetime, Array};

fn to_json(toml: Value) -> Json {
    fn doit(s: &str, json: Json) -> Json {
        let mut map = BTreeMap::new();
        map.insert(format!("{}", "type"), Json::String(format!("{}", s)));
        map.insert(format!("{}", "value"), json);
        Json::Object(map)
    }
    match toml {
        Value::String(s) => doit("string", Json::String(s)),
        Integer(i) => doit("integer", Json::String(format!("{}", i))),
        Float(f) => doit("float", Json::String({
            let s = format!("{:.15}", f);
            let s = format!("{}", s.trim_right_matches('0'));
            if s.ends_with(".") {format!("{}0", s)} else {s}
        })),
        Boolean(b) => doit("bool", Json::String(format!("{}", b))),
        Datetime(s) => doit("datetime", Json::String(s)),
        Array(arr) => {
            let is_table = match arr.first() {
                Some(&Table(..)) => true,
                _ => false,
            };
            let json = Json::Array(arr.into_iter().map(to_json).collect());
            if is_table {json} else {doit("array", json)}
        }
        Table(table) => Json::Object(table.into_iter().map(|(k, v)| {
            (k, to_json(v))
        }).collect()),
    }
}

fn run(toml: &str, json: &str) {
    let mut p = Parser::new(toml);
    let table = p.parse();
    assert!(p.errors.len() == 0, "had_errors: {:?}",
            p.errors.iter().map(|e| {
                (e.desc.clone(), &toml[e.lo - 5..e.hi + 5])
            }).collect::<Vec<(String, &str)>>());
    assert!(table.is_some());
    let toml = Table(table.unwrap());
    let toml_string = format!("{}", toml);

    let json = Json::from_str(json).unwrap();
    let toml_json = to_json(toml.clone());
    assert!(json == toml_json,
            "expected\n{}\ngot\n{}\n",
            json.pretty(),
            toml_json.pretty());

    let table2 = Parser::new(&toml_string).parse().unwrap();
    // floats are a little lossy
    if table2.values().any(|v| v.as_float().is_some()) { return }
    assert_eq!(toml, Table(table2));
}

macro_rules! test( ($name:ident, $toml:expr, $json:expr) => (
    #[test]
    fn $name() { run($toml, $json); }
) );

test!(array_empty,
       include_str!("valid/array-empty.toml"),
       include_str!("valid/array-empty.json"));
test!(array_nospaces,
       include_str!("valid/array-nospaces.toml"),
       include_str!("valid/array-nospaces.json"));
test!(arrays_hetergeneous,
       include_str!("valid/arrays-hetergeneous.toml"),
       include_str!("valid/arrays-hetergeneous.json"));
test!(arrays,
       include_str!("valid/arrays.toml"),
       include_str!("valid/arrays.json"));
test!(arrays_nested,
       include_str!("valid/arrays-nested.toml"),
       include_str!("valid/arrays-nested.json"));
test!(empty,
       include_str!("valid/empty.toml"),
       include_str!("valid/empty.json"));
test!(bool,
       include_str!("valid/bool.toml"),
       include_str!("valid/bool.json"));
test!(datetime,
       include_str!("valid/datetime.toml"),
       include_str!("valid/datetime.json"));
test!(example,
       include_str!("valid/example.toml"),
       include_str!("valid/example.json"));
test!(float,
       include_str!("valid/float.toml"),
       include_str!("valid/float.json"));
test!(implicit_and_explicit_after,
       include_str!("valid/implicit-and-explicit-after.toml"),
       include_str!("valid/implicit-and-explicit-after.json"));
test!(implicit_and_explicit_before,
       include_str!("valid/implicit-and-explicit-before.toml"),
       include_str!("valid/implicit-and-explicit-before.json"));
test!(implicit_groups,
       include_str!("valid/implicit-groups.toml"),
       include_str!("valid/implicit-groups.json"));
test!(integer,
       include_str!("valid/integer.toml"),
       include_str!("valid/integer.json"));
test!(key_equals_nospace,
       include_str!("valid/key-equals-nospace.toml"),
       include_str!("valid/key-equals-nospace.json"));
test!(key_space,
       include_str!("valid/key-space.toml"),
       include_str!("valid/key-space.json"));
test!(key_special_chars,
       include_str!("valid/key-special-chars.toml"),
       include_str!("valid/key-special-chars.json"));
test!(key_with_pound,
       include_str!("valid/key-with-pound.toml"),
       include_str!("valid/key-with-pound.json"));
test!(long_float,
       include_str!("valid/long-float.toml"),
       include_str!("valid/long-float.json"));
test!(long_integer,
       include_str!("valid/long-integer.toml"),
       include_str!("valid/long-integer.json"));
test!(multiline_string,
       include_str!("valid/multiline-string.toml"),
       include_str!("valid/multiline-string.json"));
test!(raw_multiline_string,
       include_str!("valid/raw-multiline-string.toml"),
       include_str!("valid/raw-multiline-string.json"));
test!(raw_string,
       include_str!("valid/raw-string.toml"),
       include_str!("valid/raw-string.json"));
test!(string_empty,
       include_str!("valid/string-empty.toml"),
       include_str!("valid/string-empty.json"));
test!(string_escapes,
       include_str!("valid/string-escapes.toml"),
       include_str!("valid/string-escapes.json"));
test!(string_simple,
       include_str!("valid/string-simple.toml"),
       include_str!("valid/string-simple.json"));
test!(string_with_pound,
       include_str!("valid/string-with-pound.toml"),
       include_str!("valid/string-with-pound.json"));
test!(table_array_implicit,
       include_str!("valid/table-array-implicit.toml"),
       include_str!("valid/table-array-implicit.json"));
test!(table_array_many,
       include_str!("valid/table-array-many.toml"),
       include_str!("valid/table-array-many.json"));
test!(table_array_nest,
       include_str!("valid/table-array-nest.toml"),
       include_str!("valid/table-array-nest.json"));
test!(table_array_one,
       include_str!("valid/table-array-one.toml"),
       include_str!("valid/table-array-one.json"));
test!(table_empty,
       include_str!("valid/table-empty.toml"),
       include_str!("valid/table-empty.json"));
test!(table_sub_empty,
       include_str!("valid/table-sub-empty.toml"),
       include_str!("valid/table-sub-empty.json"));
test!(table_whitespace,
       include_str!("valid/table-whitespace.toml"),
       include_str!("valid/table-whitespace.json"));
test!(table_with_pound,
       include_str!("valid/table-with-pound.toml"),
       include_str!("valid/table-with-pound.json"));
test!(unicode_escape,
       include_str!("valid/unicode-escape.toml"),
       include_str!("valid/unicode-escape.json"));
test!(unicode_literal,
       include_str!("valid/unicode-literal.toml"),
       include_str!("valid/unicode-literal.json"));
test!(hard_example,
       include_str!("valid/hard_example.toml"),
       include_str!("valid/hard_example.json"));
test!(example2,
       include_str!("valid/example2.toml"),
       include_str!("valid/example2.json"));
test!(example3,
       include_str!("valid/example-v0.3.0.toml"),
       include_str!("valid/example-v0.3.0.json"));
test!(example4,
       include_str!("valid/example-v0.4.0.toml"),
       include_str!("valid/example-v0.4.0.json"));
test!(example_bom,
       include_str!("valid/example-bom.toml"),
       include_str!("valid/example.json"));
