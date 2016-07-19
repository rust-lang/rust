// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An "interner" is a data structure that associates values with usize tags and
//! allows bidirectional lookup; i.e. given a value, one can easily find the
//! type, and vice versa.

use ast::Name;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(PartialEq, Eq, Hash)]
struct RcStr(Rc<String>);

impl Borrow<str> for RcStr {
    fn borrow(&self) -> &str {
        &self.0
    }
}

#[derive(Default)]
pub struct Interner {
    names: HashMap<RcStr, Name>,
    strings: Vec<Rc<String>>,
}

/// When traits can extend traits, we should extend index<Name,T> to get []
impl Interner {
    pub fn new() -> Self {
        Interner::default()
    }

    pub fn prefill(init: &[&str]) -> Self {
        let mut this = Interner::new();
        for &string in init {
            this.intern(string);
        }
        this
    }

    pub fn intern<T: Borrow<str> + Into<String>>(&mut self, string: T) -> Name {
        if let Some(&name) = self.names.get(string.borrow()) {
            return name;
        }

        let name = Name(self.strings.len() as u32);
        let string = Rc::new(string.into());
        self.strings.push(string.clone());
        self.names.insert(RcStr(string), name);
        name
    }

    pub fn gensym(&mut self, string: &str) -> Name {
        let gensym = Name(self.strings.len() as u32);
        // leave out of `names` to avoid colliding
        self.strings.push(Rc::new(string.to_owned()));
        gensym
    }

    /// Create a gensym with the same name as an existing entry.
    pub fn gensym_copy(&mut self, name: Name) -> Name {
        let gensym = Name(self.strings.len() as u32);
        // leave out of `names` to avoid colliding
        let string = self.strings[name.0 as usize].clone();
        self.strings.push(string);
        gensym
    }

    pub fn get(&self, name: Name) -> Rc<String> {
        self.strings[name.0 as usize].clone()
    }

    pub fn find(&self, string: &str) -> Option<Name> {
        self.names.get(string).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::Name;

    #[test]
    fn interner_tests() {
        let mut i: Interner = Interner::new();
        // first one is zero:
        assert_eq!(i.intern("dog"), Name(0));
        // re-use gets the same entry:
        assert_eq!(i.intern ("dog"), Name(0));
        // different string gets a different #:
        assert_eq!(i.intern("cat"), Name(1));
        assert_eq!(i.intern("cat"), Name(1));
        // dog is still at zero
        assert_eq!(i.intern("dog"), Name(0));
        // gensym gets 3
        assert_eq!(i.gensym("zebra"), Name(2));
        // gensym of same string gets new number :
        assert_eq!(i.gensym("zebra"), Name(3));
        // gensym of *existing* string gets new number:
        assert_eq!(i.gensym("dog"), Name(4));
        // gensym tests again with gensym_copy:
        assert_eq!(i.gensym_copy(Name(2)), Name(5));
        assert_eq!(*i.get(Name(5)), "zebra");
        assert_eq!(i.gensym_copy(Name(2)), Name(6));
        assert_eq!(*i.get(Name(6)), "zebra");
        assert_eq!(*i.get(Name(0)), "dog");
        assert_eq!(*i.get(Name(1)), "cat");
        assert_eq!(*i.get(Name(2)), "zebra");
        assert_eq!(*i.get(Name(3)), "zebra");
        assert_eq!(*i.get(Name(4)), "dog");
    }
}
