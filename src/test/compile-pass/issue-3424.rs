// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rustc --test ignores2.rs && ./ignores2

pub struct Path;

type rsrc_loader = Box<FnMut(&Path) -> Result<String, String>>;

fn tester()
{
    let mut loader: rsrc_loader = Box::new(move |_path| {
        Ok("more blah".to_string())
    });

    let path = Path;
    assert!(loader(&path).is_ok());
}

pub fn main() {}
