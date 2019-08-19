// run-pass
// Test associated types appearing in tuple-like enum variants.


use self::VarValue::*;

pub trait UnifyKey {
    type Value;
    fn to_index(&self) -> usize;
}

pub enum VarValue<K:UnifyKey> {
    Redirect(K),
    Root(K::Value, usize),
}

fn get<'a,K:UnifyKey<Value=Option<V>>,V>(table: &'a Vec<VarValue<K>>, key: &K) -> &'a Option<V> {
    match table[key.to_index()] {
        VarValue::Redirect(ref k) => get(table, k),
        VarValue::Root(ref v, _) => v,
    }
}

impl UnifyKey for usize {
    type Value = Option<char>;
    fn to_index(&self) -> usize { *self }
}

fn main() {
    let table = vec![/* 0 */ Redirect(1),
                     /* 1 */ Redirect(3),
                     /* 2 */ Root(Some('x'), 0),
                     /* 3 */ Redirect(2)];
    assert_eq!(get(&table, &0), &Some('x'));
}
