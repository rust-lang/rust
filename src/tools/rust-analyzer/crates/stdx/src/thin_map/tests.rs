#![expect(clippy::disallowed_types, reason = "test code")]

use std::collections::HashMap;

use itertools::Itertools;

use crate::thin_map::ThinMap;

macro_rules! do_with_map {
    ( $map:ident, $key:ty, $value:ty, $init_method:ident($($init_param:tt)*), $code:block $(,)? ) => {
        let map1 = {
            #[allow(unused_imports)]
            use ::std::collections::hash_map::Entry;

            #[allow(unused_mut)]
            let mut $map = ::std::collections::HashMap::<$key, $value>::$init_method($($init_param)*);
            $code;
            $map
        };
        let map2 = {
            #[allow(unused_imports)]
            use super::Entry;

            #[allow(unused_mut)]
            let mut $map = super::ThinMap::<$key, $value, ::std::collections::HashMap::<$key, $value>>::$init_method($($init_param)*);
            $code;
            $map
        };
        let map1 = map1.into_iter().sorted_unstable().collect::<Vec<_>>();
        let map2 = map2.into_iter().sorted_unstable().collect::<Vec<_>>();
        assert_eq!(map1, map2);
    };
}

#[test]
fn basic() {
    do_with_map! {
        map, i32, i32, new(),
        {
            for key in 0..10 {
                assert_eq!(map.insert(key, key), None);
            }
            assert_eq!(map.insert(0, 123), Some(0));
            assert_eq!(map.get(&0), Some(&123));
            for (key, value) in &map {
                assert!(key == value || *key == 0);
            }
            assert_eq!(map.len(), 10);
            assert_eq!(map.remove(&1), Some(1));
            assert_eq!(map.len(), 9);
        }
    }
    do_with_map! {
        map, i32, i32, new(),
        {
            for key in 0..100 {
                assert_eq!(map.insert(key, key), None);
            }
            for key in 5..100 {
                assert!(map.remove(&key).is_some());
            }
            map.shrink_to_fit();
            assert_eq!(map.len(), 5);
            for key in 0..5 {
                *map.get_mut(&key).unwrap() = -111;
            }
            for key in 0..5 {
                assert!(map.remove(&key).is_some());
            }
            assert!(map.is_empty());
            map.shrink_to_fit();
        }
    }
    do_with_map! {
        map, i32, i32, with_capacity(100),
        {
        }
    }
    do_with_map! {
        map, i32, i32, with_capacity(5),
        {
        }
    }
    do_with_map! {
        map, i32, i32, with_capacity(5),
        {
            map.reserve(100);
        }
    }
}

#[test]
fn with_drop_type() {
    do_with_map! {
        map, String, String, new(),
        {
            for key in 0..10 {
                assert_eq!(map.insert(key.to_string(), key.to_string()), None);
            }
            assert_eq!(map.insert(0.to_string(), 123.to_string()), Some(0.to_string()));
            assert_eq!(map.get(&0.to_string()), Some(&123.to_string()));
            for (key, value) in &map {
                assert!(key == value || *key == "0");
            }
            assert_eq!(map.len(), 10);
            assert_eq!(map.remove(&1.to_string()), Some(1.to_string()));
            assert_eq!(map.len(), 9);
        }
    }
    do_with_map! {
        map, String, String, new(),
        {
            for key in 0..100 {
                assert_eq!(map.insert(key.to_string(), key.to_string()), None);
            }
            for key in 5..100 {
                assert!(map.remove(&key.to_string()).is_some());
            }
            map.shrink_to_fit();
            assert_eq!(map.len(), 5);
            for key in 0..5 {
                *map.get_mut(&key.to_string()).unwrap() = String::new();
            }
            for key in 0..5 {
                assert!(map.remove(&key.to_string()).is_some());
            }
            assert!(map.is_empty());
            map.shrink_to_fit();
        }
    }
    do_with_map! {
        map, String, String, with_capacity(100),
        {
        }
    }
    do_with_map! {
        map, String, String, with_capacity(5),
        {
        }
    }
    do_with_map! {
        map, String, String, with_capacity(5),
        {
            map.reserve(100);
        }
    }
}

#[test]
fn with_reference_aliasing() {
    {
        let mut data = Vec::from_iter(0..20);
        let mut map = ThinMap::<&mut i32, &mut i32, HashMap<&mut i32, &mut i32>>::new();
        for [key, value] in data.as_chunks_mut().0 {
            assert_eq!(map.insert(key, value), None);
        }
        assert_eq!(map.len(), 10);
        for (_key, _value) in &map {}
        for (_key, _value) in &mut map {}
        for (_key, _value) in map {}
    }
    let data = Vec::from_iter(0..200);
    do_with_map! {
        map, &i32, &i32, new(),
        {
            for key in &data[0..100] {
                assert_eq!(map.insert(key, key), None);
            }
            for key in &data[5..100] {
                assert!(map.remove(key).is_some());
            }
            map.shrink_to_fit();
            assert_eq!(map.len(), 5);
            for key in 0..5 {
                *map.get_mut(&data[key]).unwrap() = &data[150];
            }
            for key in 0..5 {
                assert!(map.remove(&key).is_some());
            }
            assert!(map.is_empty());
            map.shrink_to_fit();
        }
    }
    do_with_map! {
        map, &i32, &i32, with_capacity(100),
        {
        }
    }
    do_with_map! {
        map, &i32, &i32, with_capacity(5),
        {
        }
    }
    do_with_map! {
        map, &mut i32, &mut i32, with_capacity(5),
        {
            map.reserve(100);
        }
    }
}

#[test]
fn entry_api() {
    do_with_map! {
        map, String, String, new(),
        {
            match map.entry("abc".to_owned()) {
                Entry::Occupied(_) => panic!("should be vacant"),
                Entry::Vacant(entry) => entry.insert(String::new()),
            };
            match map.entry("abc".to_owned()) {
                Entry::Vacant(_) => panic!("should be occupied"),
                Entry::Occupied(mut entry) => {
                    assert_eq!(entry.get(), "");
                    entry.insert("def".to_owned());
                }
            }
            assert_eq!(map.get("abc"), Some(&"def".to_owned()));
        }
    }
    do_with_map! {
        map, i32, i32, new(),
        {
            for key in 0..100 {
                map.insert(key, 0);
            }
            match map.entry(1000) {
                Entry::Occupied(_) => panic!("should be vacant"),
                Entry::Vacant(entry) => entry.insert(0),
            };
            match map.entry(1000) {
                Entry::Vacant(_) => panic!("should be occupied"),
                Entry::Occupied(mut entry) => {
                    assert_eq!(*entry.get(), 0);
                    entry.insert(123456);
                }
            }
            assert_eq!(map.get(&1000), Some(&123456));
        }
    }
}
