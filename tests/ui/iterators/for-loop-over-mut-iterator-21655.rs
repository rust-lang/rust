// https://github.com/rust-lang/rust/issues/21655
//@ run-pass

fn test(it: &mut dyn Iterator<Item=i32>) {
    for x in it {
        assert_eq!(x, 1)
    }
}

fn main() {
    let v = vec![1];
    test(&mut v.into_iter())
}
