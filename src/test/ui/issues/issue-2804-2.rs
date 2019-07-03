// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// Minimized version of issue-2804.rs. Both check that callee IDs don't
// clobber the previous node ID in a macro expr

use std::collections::HashMap;

fn add_interfaces(managed_ip: String, device: HashMap<String, isize>)  {
     println!("{}, {}", managed_ip, device["interfaces"]);
}

pub fn main() {}
