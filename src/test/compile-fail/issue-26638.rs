// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn parse_type(iter: Box<Iterator<Item=&str>+'static>) -> &str { iter.next() }
//~^ ERROR missing lifetime specifier [E0106]
//~^^ HELP 2 elided lifetimes

fn parse_type_2(iter: fn(&u8)->&u8) -> &str { iter() }
//~^ ERROR missing lifetime specifier [E0106]
//~^^ HELP lifetime cannot be derived

fn parse_type_3() -> &str { unimplemented!() }
//~^ ERROR missing lifetime specifier [E0106]
//~^^ HELP no value for it to be borrowed from

fn main() {}
