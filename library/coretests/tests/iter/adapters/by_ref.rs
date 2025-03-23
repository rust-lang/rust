#[test]
fn test_iterator_by_ref_sized() {
    let a = ['a', 'b', 'c', 'd'];

    let mut s = String::from("Z");
    let mut it = a.iter().copied();
    (&mut it).take(2).for_each(|x| s.push(x));
    assert_eq!(s, "Zab");
    (&mut it).fold((), |(), x| s.push(x));
    assert_eq!(s, "Zabcd");

    let mut s = String::from("Z");
    let mut it = a.iter().copied();
    (&mut it).rev().take(2).for_each(|x| s.push(x));
    assert_eq!(s, "Zdc");
    (&mut it).rfold((), |(), x| s.push(x));
    assert_eq!(s, "Zdcba");
}
