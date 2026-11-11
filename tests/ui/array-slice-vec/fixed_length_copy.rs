//@ run-pass


pub fn main() {
    let arr = [1,2,3];
    let arr2 = arr;
    assert_eq!(arr[1], 2);
    assert_eq!(arr2[2], 3);
}
