// run-pass
pub fn main() {
    let foo = (Some(1), (), (), vec![2, 3]);

    match &foo {
        (Some(n), .., v) => {
            assert_eq!((*v).len(), 2);
            assert_eq!(*n, 1);
        }
        (None, (), (), ..) => panic!(),
    }
}
