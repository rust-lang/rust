// run-pass


pub fn main() {
    let  v = vec![1,2,3,4,5];
    let v2 = &v[1..3];
    assert_eq!(v2[0], 2);
    assert_eq!(v2[1], 3);
}
