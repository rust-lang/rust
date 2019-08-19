// run-pass

fn pairwise_sub<T:DoubleEndedIterator<Item=isize>>(mut t: T) -> isize {
    let mut result = 0;
    loop {
        let front = t.next();
        let back = t.next_back();
        match (front, back) {
            (Some(f), Some(b)) => { result += b - f; }
            _ => { return result; }
        }
    }
}

fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];
    let r = pairwise_sub(v.into_iter());
    assert_eq!(r, 9);
}
