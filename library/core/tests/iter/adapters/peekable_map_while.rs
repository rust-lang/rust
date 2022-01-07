use core::iter::*;

#[test]
fn test_iterator_peekable_map_while() {
    let vec = vec!["0", "1", "2", "three", "four"];
    let mut xs = vec.iter().peekable();
    let ys: Vec<u8> = xs.peekable_map_while(|x| x.parse().ok()).collect();

    assert_eq!(ys, vec![0, 1, 2]);
    assert_eq!(xs.next(), Some(&"three"));
    assert_eq!(xs.next(), Some(&"four"));
    assert_eq!(xs.next(), None);

    let vec = vec!["0", "1", "2", "3", "4"];
    let mut xs = vec.iter().peekable();
    let ys: Vec<u8> = xs.peekable_map_while(|x| x.parse().ok()).collect();

    assert_eq!(ys, vec![0, 1, 2, 3, 4]);
    assert_eq!(xs.next(), None);

    let vec = vec!["zero", "one", "two", "three", "four"];
    let mut xs = vec.iter().peekable();
    let ys: Vec<u8> = xs.peekable_map_while(|x| x.parse().ok()).collect();

    assert_eq!(ys, vec![]);
    assert_eq!(xs.next(), Some(&"zero"));
    assert_eq!(xs.next(), Some(&"one"));
    assert_eq!(xs.next(), Some(&"two"));
    assert_eq!(xs.next(), Some(&"three"));
    assert_eq!(xs.next(), Some(&"four"));
    assert_eq!(xs.next(), None);
}

#[test]
fn test_iterator_peekable_map_while_fold() {
    let mut xs = ["0", "1", "2", "3", "4", "5", "six"].iter().peekable();
    let ys = [0, 5, 10, 15, 20, 25];
    let it = xs.peekable_map_while(|x| x.parse::<usize>().ok().map(|x| x * 5));
    let i = it.fold(0, |i, x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());
    assert_eq!(xs.next(), Some(&"six"));
    assert_eq!(xs.next(), None);
}
