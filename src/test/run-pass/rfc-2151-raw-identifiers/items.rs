// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(raw_identifiers)]

#[derive(Debug, PartialEq, Eq)]
struct IntWrapper(u32);

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Hash, Copy, Clone, Default)]
struct HasKeywordField {
    r#struct: u32,
}

struct Generic<r#T>(T);

trait Trait {
    fn r#trait(&self) -> u32;
}
impl Trait for Generic<u32> {
    fn r#trait(&self) -> u32 {
        self.0
    }
}

pub fn main() {
    assert_eq!(IntWrapper(1), r#IntWrapper(1));

    match IntWrapper(2) {
        r#IntWrapper(r#struct) => assert_eq!(2, r#struct),
    }

    assert_eq!("HasKeywordField { struct: 3 }", format!("{:?}", HasKeywordField { r#struct: 3 }));

    assert_eq!(4, Generic(4).0);
    assert_eq!(5, Generic(5).r#trait());
}
