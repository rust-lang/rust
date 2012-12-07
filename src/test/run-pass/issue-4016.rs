// xfail-test
extern mod std;

use send_map::linear;
use std::json;
use std::serialization::{Deserializable, deserialize};

trait JD : Deserializable<json::Deserializer> { }
//type JD = Deserializable<json::Deserializer>;

fn exec<T: JD>() {
    let doc = result::unwrap(json::from_str(""));
    let _v: T = deserialize(&json::Deserializer(move doc));
    fail
}

fn main() {}
