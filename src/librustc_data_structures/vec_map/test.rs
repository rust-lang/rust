use super::*;

type Element = (usize, &'static str);

fn test_map() -> VecMap<Element, impl Fn(&Element) -> usize> {
    let data = vec![(3, "three-a"), (0, "zero"), (3, "three-b"), (22, "twenty-two")];
    VecMap::new(data, |&(key, _)| key)
}

#[test]
fn get_first() {
    let map = test_map();
    assert_eq!(map.get_first(&0), Some(&(0, "zero")));
    assert_eq!(map.get_first(&1), None);
    assert_eq!(map.get_first(&3), Some(&(3, "three-a")));
    assert_eq!(map.get_first(&22), Some(&(22, "twenty-two")));
    assert_eq!(map.get_first(&23), None);
}

#[test]
fn get_all() {
    let map = test_map();
    assert_eq!(map.get_all(&0), &[(0, "zero")]);
    assert_eq!(map.get_all(&1), &[]);
    assert_eq!(map.get_all(&3), &[(3, "three-a"), (3, "three-b")]);
    assert_eq!(map.get_all(&22), &[(22, "twenty-two")]);
    assert_eq!(map.get_all(&23), &[]);
}
