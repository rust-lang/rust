// run-pass



pub fn main() {
    assert_eq!(vec![1; 3], vec![1, 1, 1]);
    assert_eq!(vec![1; 2], vec![1, 1]);
    assert_eq!(vec![1; 1], vec![1]);
    assert_eq!(vec![1; 0], vec![]);

    // from_elem syntax (see RFC 832)
    let el = Box::new(1);
    let n = 3;
    assert_eq!(vec![el; n], vec![Box::new(1), Box::new(1), Box::new(1)]);
}
