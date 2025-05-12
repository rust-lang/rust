//@ run-pass
#![allow(unused_variables)]
//@ aux-build:issue-19340-1.rs


extern crate issue_19340_1 as lib;

use lib::Homura;

fn main() {
    let homura = Homura::Madoka { name: "Kaname".to_string() };

    match homura {
        Homura::Madoka { name } => (),
    };
}
