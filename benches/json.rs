#![feature(test)]

extern crate test;
extern crate rustc_serialize;

use std::string;
use rustc_serialize::json::{Json, Parser};
use test::Bencher;

#[bench]
fn bench_streaming_small(b: &mut Bencher) {
    b.iter( || {
        let mut parser = Parser::new(
            r#"{
                "a": 1.0,
                "b": [
                    true,
                    "foo\nbar",
                    { "c": {"d": null} }
                ]
            }"#.chars()
        );
        loop {
            match parser.next() {
                None => return,
                _ => {}
            }
        }
    });
}
#[bench]
fn bench_small(b: &mut Bencher) {
    b.iter( || {
        let _ = Json::from_str(r#"{
            "a": 1.0,
            "b": [
                true,
                "foo\nbar",
                { "c": {"d": null} }
            ]
        }"#);
    });
}

#[bench]
fn bench_decode_hex_escape(b: &mut Bencher) {
    let mut src = "\"".to_string();
    for _ in 0..10 {
        src.push_str("\\uF975\\uf9bc\\uF9A0\\uF9C4\\uF975\\uf9bc\\uF9A0\\uF9C4");
    }
    src.push_str("\"");
    b.iter( || {
        let _ = Json::from_str(&src);
    });
}

fn big_json() -> string::String {
    let mut src = "[\n".to_string();
    for _ in 0..500 {
        src.push_str(r#"{ "a": true, "b": null, "c":3.1415, "d": "Hello world", "e": \
                        [1,2,3]},"#);
    }
    src.push_str("{}]");
    return src;
}

#[bench]
fn bench_streaming_large(b: &mut Bencher) {
    let src = big_json();
    b.iter( || {
        let mut parser = Parser::new(src.chars());
        loop {
            match parser.next() {
                None => return,
                _ => {}
            }
        }
    });
}
#[bench]
fn bench_large(b: &mut Bencher) {
    let src = big_json();
    b.iter( || { let _ = Json::from_str(&src); });
}
