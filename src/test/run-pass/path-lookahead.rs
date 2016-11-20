// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Parser test for #37765

fn with_parens<T: ToString>(arg: T) -> String { //~WARN dead_code
  return (<T as ToString>::to_string(&arg)); //~WARN unused_parens
}

fn no_parens<T: ToString>(arg: T) -> String { //~WARN dead_code
  return <T as ToString>::to_string(&arg);
}

fn main() {

}
