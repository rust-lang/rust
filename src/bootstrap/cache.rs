// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use serde_json;
use serde::{Serialize, Deserialize};

use std::fmt;
use std::mem;
use std::intrinsics;
use std::collections::HashMap;
use std::cell::RefCell;

/// This is essentially a HashMap which allows storing any type in its input and
/// any type in its output. It is a write-once cache; values are never evicted,
/// which means that references to the value can safely be returned from the
/// get() method.
//
// FIXME: This type does not permit retrieving &Path from a PathBuf, primarily
// due to a lack of any obvious way to ensure that this is safe, but also not
// penalize other cases (e.g., deserializing u32 -> &u32, which is non-optimal).
#[derive(Debug)]
pub struct Cache(RefCell<HashMap<Key, Box<str>>>);

fn to_json<T: Serialize>(element: &T) -> String {
    let type_id = unsafe {
        intrinsics::type_name::<T>()
    };

    t!(serde_json::to_string(&(type_id, element)))
}

fn from_json<'a, O: Deserialize<'a>>(data: &'a str) -> O {
    let type_id = unsafe {
        intrinsics::type_name::<O>()
    };

    let (de_type_id, element): (&'a str, O)  = t!(serde_json::from_str(data));

    assert_eq!(type_id, de_type_id);

    element
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Key(String);

impl fmt::Debug for Key {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(&self.0)
    }
}

impl Cache {
    pub fn new() -> Cache {
        Cache(RefCell::new(HashMap::new()))
    }

    pub fn to_key<K: Serialize>(key: &K) -> Key {
        Key(to_json(key))
    }

    /// Puts a value into the cache. Will panic if called more than once with
    /// the same key.
    ///
    /// Returns the internal key utilized, as an opaque structure, useful only
    /// for debugging.
    pub fn put<V>(&self, key: Key, value: &V)
    where
        V: Serialize,
    {
        let mut cache = self.0.borrow_mut();
        let value = to_json(value);
        assert!(!cache.contains_key(&key), "processing {:?} a second time", key);
        // Store a boxed str so that it's location in memory never changes and
        // it's safe for us to return references to it (so long as they live as
        // long as us).
        cache.insert(key, value.into_boxed_str());
    }

    pub fn get<'a, V>(&'a self, key: &Key) -> Option<V>
    where
        V: Deserialize<'a> + 'a,
    {
        let cache = self.0.borrow();
        cache.get(key).map(|v| {
            // Change the lifetime. This borrow is valid for as long as self lives;
            // the data we're accessing will live as long as us and will be in a
            // stable location, since we use Box<str>.
            let v = unsafe {
                mem::transmute::<&str, &'a str>(v)
            };
            from_json(v)
        })
    }
}
