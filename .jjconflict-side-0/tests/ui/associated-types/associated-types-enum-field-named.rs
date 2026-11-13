//@ run-pass
// Test associated types appearing in struct-like enum variants.


use self::VarValue::*;

pub trait UnifyKey {
    type Value;
    fn to_index(&self) -> usize;
}

pub enum VarValue<K:UnifyKey> {
    Redirect { to: K },
    Root { value: K::Value, rank: usize },
}

fn get<'a,K:UnifyKey<Value=Option<V>>,V>(table: &'a Vec<VarValue<K>>, key: &K) -> &'a Option<V> {
    match table[key.to_index()] {
        VarValue::Redirect { to: ref k } => get(table, k),
        VarValue::Root { value: ref v, rank: _ } => v,
    }
}

impl UnifyKey for usize {
    type Value = Option<char>;
    fn to_index(&self) -> usize { *self }
}

fn main() {
    let table = vec![/* 0 */ Redirect { to: 1 },
                     /* 1 */ Redirect { to: 3 },
                     /* 2 */ Root { value: Some('x'), rank: 0 },
                     /* 3 */ Redirect { to: 2 }];
    assert_eq!(get(&table, &0), &Some('x'));
}
