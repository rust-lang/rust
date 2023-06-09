// run-pass
use std::collections::BTreeMap;
use std::borrow::Cow;

use std::borrow::Cow::{Owned as O, Borrowed as B};

type SendStr = Cow<'static, str>;

fn main() {
    let mut map: BTreeMap<SendStr, usize> = BTreeMap::new();
    assert!(map.insert(B("foo"), 42).is_none());
    assert!(map.insert(O("foo".to_string()), 42).is_some());
    assert!(map.insert(B("foo"), 42).is_some());
    assert!(map.insert(O("foo".to_string()), 42).is_some());

    assert!(map.insert(B("foo"), 43).is_some());
    assert!(map.insert(O("foo".to_string()), 44).is_some());
    assert!(map.insert(B("foo"), 45).is_some());
    assert!(map.insert(O("foo".to_string()), 46).is_some());

    let v = 46;

    assert_eq!(map.get(&O("foo".to_string())), Some(&v));
    assert_eq!(map.get(&B("foo")), Some(&v));

    let (a, b, c, d) = (50, 51, 52, 53);

    assert!(map.insert(B("abc"), a).is_none());
    assert!(map.insert(O("bcd".to_string()), b).is_none());
    assert!(map.insert(B("cde"), c).is_none());
    assert!(map.insert(O("def".to_string()), d).is_none());

    assert!(map.insert(B("abc"), a).is_some());
    assert!(map.insert(O("bcd".to_string()), b).is_some());
    assert!(map.insert(B("cde"), c).is_some());
    assert!(map.insert(O("def".to_string()), d).is_some());

    assert!(map.insert(O("abc".to_string()), a).is_some());
    assert!(map.insert(B("bcd"), b).is_some());
    assert!(map.insert(O("cde".to_string()), c).is_some());
    assert!(map.insert(B("def"), d).is_some());

    assert_eq!(map.get(&B("abc")), Some(&a));
    assert_eq!(map.get(&B("bcd")), Some(&b));
    assert_eq!(map.get(&B("cde")), Some(&c));
    assert_eq!(map.get(&B("def")), Some(&d));

    assert_eq!(map.get(&O("abc".to_string())), Some(&a));
    assert_eq!(map.get(&O("bcd".to_string())), Some(&b));
    assert_eq!(map.get(&O("cde".to_string())), Some(&c));
    assert_eq!(map.get(&O("def".to_string())), Some(&d));

    assert!(map.remove(&B("foo")).is_some());
    assert_eq!(map.into_iter().map(|(k, v)| format!("{}{}", k, v))
                              .collect::<Vec<String>>()
                              .concat(),
               "abc50bcd51cde52def53".to_string());
}
