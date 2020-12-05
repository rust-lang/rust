// run-pass


fn swap<F>(f: F) -> Vec<isize> where F: FnOnce(Vec<isize>) -> Vec<isize> {
    let x = vec![1, 2, 3];
    f(x)
}

pub fn main() {
    let v = swap(|mut x| { x.push(4); x });
    let w = swap(|mut x| { x.push(4); x });
    assert_eq!(v, w);
}
