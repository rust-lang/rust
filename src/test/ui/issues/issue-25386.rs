// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod stuff {
    pub struct Item {
        c_object: Box<CObj>,
    }
    pub struct CObj {
        name: Option<String>,
    }
    impl Item {
        pub fn new() -> Item {
            Item {
                c_object: Box::new(CObj { name: None }),
            }
        }
    }
}

macro_rules! check_ptr_exist {
    ($var:expr, $member:ident) => (
        (*$var.c_object).$member.is_some()
        //~^ ERROR field `name` of struct `stuff::CObj` is private
        //~^^ ERROR field `c_object` of struct `stuff::Item` is private
    );
}

fn main() {
    let item = stuff::Item::new();
    println!("{}", check_ptr_exist!(item, name));
}
