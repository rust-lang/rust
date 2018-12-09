// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Copy, Clone)]
struct HasChars;

impl HasChars {
    fn chars(self) -> std::str::Chars<'static> {
        "HasChars".chars()
    }
}

fn main() {
    let abc = "abc";
    let def = String::from("def");
    let mut s = String::new();

    s.push_str(abc);
    s.extend(abc.chars());

    s.push_str("abc");
    s.extend("abc".chars());

    s.push_str(&def);
    s.extend(def.chars());

    s.extend(abc.chars().skip(1));
    s.extend("abc".chars().skip(1));
    s.extend(['a', 'b', 'c'].iter());

    let f = HasChars;
    s.extend(f.chars());
}
