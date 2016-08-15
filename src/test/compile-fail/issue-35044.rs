// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Debug, PartialEq)]
enum Reg {
    EAX,
    EBX,
    ECX,
    EDX,
    ESP,
    EBP,
    ISP,
}

fn string_to_reg(_s:&str) -> Reg {
    match _s.as_ref() {
        "EAX" => Reg::EAX,
        "EBX" => Reg::EBX,
        "ECX" => Reg::ECX,
        "EDX" => Reg::EDX,
        "EBP" => Reg::EBP,
        "ESP" => Reg::ESP,
        "ISP" => Reg::ISP, //~ NOTE split literal here
        &_    => panic!("bla bla bla"), //~ ERROR see issue #35044
    }
}

fn main() {
    let vec = vec!["EAX", "EBX", "ECX", "EDX", "ESP", "EBP", "ISP"];
    let mut iter = vec.iter();
    println!("{:?}", string_to_reg(""));
    println!("{:?}", string_to_reg(iter.next().unwrap()));
    println!("{:?}", string_to_reg(iter.next().unwrap()));
    println!("{:?}", string_to_reg(iter.next().unwrap()));
    println!("{:?}", string_to_reg(iter.next().unwrap()));
    println!("{:?}",  string_to_reg(iter.next().unwrap()));
    println!("{:?}", string_to_reg(iter.next().unwrap()));
    println!("{:?}",  string_to_reg(iter.next().unwrap()));
}
