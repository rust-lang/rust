//@ run-rustfix
#![allow(unused_variables, dead_code)]
use std::collections::BTreeMap;
use std::collections::HashSet;

#[derive(Debug,Eq,PartialEq,Hash)]
enum Day {
    Mon,
}

struct Class {
    days: BTreeMap<u32, HashSet<Day>>
}

impl Class {
    fn do_stuff(&self) {
        for (_, v) in &self.days {
            let mut x: HashSet<Day> = v.clone(); //~ ERROR
            let y: Vec<Day> = x.drain().collect();
            println!("{:?}", x);
        }
    }
}

fn fail() {
    let c = Class { days: BTreeMap::new() };
    c.do_stuff();
}
fn main() {}
