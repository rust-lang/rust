use super::*;

impl<K, V> VecMap<K, V> {
    fn into_vec(self) -> Vec<(K, V)> {
        self.0.into()
    }
}

#[test]
fn test_from_iterator() {
    assert_eq!(
        std::iter::empty().collect::<VecMap<i32, bool>>().into_vec(),
        Vec::<(i32, bool)>::new()
    );
    assert_eq!(std::iter::once((42, true)).collect::<VecMap<_, _>>().into_vec(), vec![(42, true)]);
    assert_eq!(
        [(1, true), (2, false)].into_iter().collect::<VecMap<_, _>>().into_vec(),
        vec![(1, true), (2, false)]
    );
}

#[test]
fn test_into_iterator_owned() {
    assert_eq!(VecMap::new().into_iter().collect::<Vec<(i32, bool)>>(), Vec::<(i32, bool)>::new());
    assert_eq!(VecMap::from(vec![(1, true)]).into_iter().collect::<Vec<_>>(), vec![(1, true)]);
    assert_eq!(
        VecMap::from(vec![(1, true), (2, false)]).into_iter().collect::<Vec<_>>(),
        vec![(1, true), (2, false)]
    );
}

#[test]
fn test_insert() {
    let mut v = VecMap::new();
    assert_eq!(v.insert(1, true), None);
    assert_eq!(v.insert(2, false), None);
    assert_eq!(v.clone().into_vec(), vec![(1, true), (2, false)]);
    assert_eq!(v.insert(1, false), Some(true));
    assert_eq!(v.into_vec(), vec![(1, false), (2, false)]);
}

#[test]
fn test_get() {
    let v = [(1, true), (2, false)].into_iter().collect::<VecMap<_, _>>();
    assert_eq!(v.get(&1), Some(&true));
    assert_eq!(v.get(&2), Some(&false));
    assert_eq!(v.get(&3), None);
}
