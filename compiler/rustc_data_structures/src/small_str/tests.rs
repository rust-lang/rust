use super::*;

#[test]
fn empty() {
    let s = SmallStr::<1>::new();
    assert!(s.empty());
    assert_eq!("", s.as_str());
    assert!(!s.spilled());
}

#[test]
fn from_iter() {
    let s = ["aa", "bb", "cc"].iter().collect::<SmallStr<6>>();
    assert_eq!("aabbcc", s.as_str());
    assert!(!s.spilled());

    let s = ["aa", "bb", "cc", "dd"].iter().collect::<SmallStr<6>>();
    assert_eq!("aabbccdd", s.as_str());
    assert!(s.spilled());
}
