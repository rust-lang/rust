// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:--test -O

#[test]
#[should_panic(expected = "creating inhabited type")]
fn test() {
    FontLanguageOverride::system_font(SystemFont::new());
}

pub enum FontLanguageOverride {
    Normal,
    Override(&'static str),
    System(SystemFont)
}

pub enum SystemFont {}

impl FontLanguageOverride {
    fn system_font(f: SystemFont) -> Self {
        FontLanguageOverride::System(f)
    }
}

impl SystemFont {
    fn new() -> Self {
        panic!("creating inhabited type")
    }
}
