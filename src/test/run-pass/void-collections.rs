// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashMap;
use std::collections::BTreeMap;

#[derive(Eq, PartialEq, Hash, PartialOrd, Ord)]
enum Void {}

trait Foo {}

impl<T> Foo for T {}

fn main() {
    std::rc::Weak::<Void>::new();
    std::rc::Weak::<Void>::new().clone();
    (std::rc::Weak::<Void>::new() as std::rc::Weak<Foo>);
    (std::rc::Weak::<Void>::new() as std::rc::Weak<Foo>).clone();
    std::sync::Weak::<Void>::new();
    (std::sync::Weak::<Void>::new() as std::sync::Weak<Foo>);
    (std::sync::Weak::<Void>::new() as std::sync::Weak<Foo>).clone();

    let mut h: HashMap<Void, Void> = HashMap::new();
    assert_eq!(h.len(), 0);
    assert_eq!(h.iter().count(), 0);
    assert_eq!(h.iter_mut().count(), 0);
    assert_eq!(h.into_iter().count(), 0);

    let mut h: BTreeMap<Void, Void> = BTreeMap::new();
    assert_eq!(h.len(), 0);
    assert_eq!(h.iter().count(), 0);
    assert_eq!(h.iter_mut().count(), 0);
    assert_eq!(h.into_iter().count(), 0);

    let mut h: Vec<Void> = Vec::new();
    assert_eq!(h.len(), 0);
    assert_eq!(h.iter().count(), 0);
    assert_eq!(h.iter_mut().count(), 0);
    assert_eq!(h.into_iter().count(), 0);
}
