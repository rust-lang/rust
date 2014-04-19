
// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate collections;
extern crate serialize;

use collections::HashMap;
use serialize::json;
use std::option;

enum object {
    bool_value(bool),
    int_value(i64),
}

fn lookup(table: ~json::Object, key: ~str, default: ~str) -> ~str
{
    match table.find(&key) {
        option::Some(&json::String(ref s)) => {
            (*s).clone()
        }
        option::Some(value) => {
            println!("{} was expected to be a string but is a {:?}", key, value);
            default
        }
        option::None => {
            default
        }
    }
}

fn add_interface(_store: int, managed_ip: ~str, data: json::Json) -> (~str, object)
{
    match &data {
        &json::Object(ref interface) => {
            let name = lookup((*interface).clone(), "ifDescr".to_owned(), "".to_owned());
            let label = format!("{}-{}", managed_ip, name);

            (label, bool_value(false))
        }
        _ => {
            println!("Expected dict for {} interfaces but found {:?}", managed_ip, data);
            ("gnos:missing-interface".to_owned(), bool_value(true))
        }
    }
}

fn add_interfaces(store: int, managed_ip: ~str, device: HashMap<~str, json::Json>)
-> Vec<(~str, object)> {
    match device.get(&"interfaces".to_owned())
    {
        &json::List(ref interfaces) =>
        {
          interfaces.iter().map(|interface| {
                add_interface(store, managed_ip.clone(), (*interface).clone())
          }).collect()
        }
        _ =>
        {
            println!("Expected list for {} interfaces but found {:?}", managed_ip,
                   device.get(&"interfaces".to_owned()));
            Vec::new()
        }
    }
}

pub fn main() {}
