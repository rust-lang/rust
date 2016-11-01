#![deny(warnings)]

extern crate toml;
extern crate rustc_serialize;

use std::fs::File;
use std::env;
use std::io;
use std::io::prelude::*;

use toml::Value;
use rustc_serialize::json::Json;

fn main() {
    let mut args = env::args();
    let mut input = String::new();
    let filename = if args.len() > 1 {
        let name = args.nth(1).unwrap();
        File::open(&name).and_then(|mut f| {
            f.read_to_string(&mut input)
        }).unwrap();
        name
    } else {
        io::stdin().read_to_string(&mut input).unwrap();
        "<stdin>".to_string()
    };

    let mut parser = toml::Parser::new(&input);
    let toml = match parser.parse() {
        Some(toml) => toml,
        None => {
            for err in &parser.errors {
                let (loline, locol) = parser.to_linecol(err.lo);
                let (hiline, hicol) = parser.to_linecol(err.hi);
                println!("{}:{}:{}-{}:{} error: {}",
                         filename, loline, locol, hiline, hicol, err.desc);
            }
            return
        }
    };
    let json = convert(Value::Table(toml));
    println!("{}", json.pretty());
}

fn convert(toml: Value) -> Json {
    match toml {
        Value::String(s) => Json::String(s),
        Value::Integer(i) => Json::I64(i),
        Value::Float(f) => Json::F64(f),
        Value::Boolean(b) => Json::Boolean(b),
        Value::Array(arr) => Json::Array(arr.into_iter().map(convert).collect()),
        Value::Table(table) => Json::Object(table.into_iter().map(|(k, v)| {
            (k, convert(v))
        }).collect()),
        Value::Datetime(dt) => Json::String(dt),
    }
}
