// run-pass



pub fn main() {
    let mut v = vec![1];
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);
    assert_eq!(v[0], 1);
    assert_eq!(v[1], 2);
    assert_eq!(v[2], 3);
    assert_eq!(v[3], 4);
    assert_eq!(v[4], 5);
}
