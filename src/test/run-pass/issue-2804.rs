// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

use extra::json;
use std::hashmap::HashMap;
use std::option;

enum object {
    bool_value(bool),
    int_value(i64),
}

fn lookup(table: ~json::Object, key: ~str, default: ~str) -> ~str
{
    match table.find(&key) {
        option::Some(&extra::json::String(ref s)) => {
            (*s).clone()
        }
        option::Some(value) => {
            error2!("{} was expected to be a string but is a {:?}", key, value);
            default
        }
        option::None => {
            default
        }
    }
}

fn add_interface(_store: int, managed_ip: ~str, data: extra::json::Json) -> (~str, object)
{
    match &data {
        &extra::json::Object(ref interface) => {
            let name = lookup((*interface).clone(), ~"ifDescr", ~"");
            let label = format!("{}-{}", managed_ip, name);

            (label, bool_value(false))
        }
        _ => {
            error2!("Expected dict for {} interfaces but found {:?}", managed_ip, data);
            (~"gnos:missing-interface", bool_value(true))
        }
    }
}

fn add_interfaces(store: int, managed_ip: ~str, device: HashMap<~str, extra::json::Json>) -> ~[(~str, object)]
{
    match device.get(&~"interfaces")
    {
        &extra::json::List(ref interfaces) =>
        {
          do interfaces.map |interface| {
                add_interface(store, managed_ip.clone(), (*interface).clone())
          }
        }
        _ =>
        {
            error2!("Expected list for {} interfaces but found {:?}", managed_ip,
                   device.get(&~"interfaces"));
            ~[]
        }
    }
}

pub fn main() {}
