// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


pub struct CrateId {
    local_path: String,
    junk: String
}

impl CrateId {
    fn new(s: &str) -> CrateId {
        CrateId {
            local_path: s.to_string(),
            junk: "wutevs".to_string()
        }
    }
}

pub fn remove_package_from_database() {
    let mut lines_to_use: Vec<&CrateId> = Vec::new();
    let push_id = |installed_id: &CrateId| {
        lines_to_use.push(installed_id);
        //~^ ERROR cannot infer an appropriate lifetime for automatic coercion due to
        // conflicting requirements
    };
    list_database(push_id);

    for l in lines_to_use.iter() {
        println!("{}", l.local_path);
    }

}

pub fn list_database(f: |&CrateId|) {
    let stuff = ["foo", "bar"];

    for l in stuff.iter() {
        f(&CrateId::new(*l));
    }
}

pub fn main() {
    remove_package_from_database();
}
