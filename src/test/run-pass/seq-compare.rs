pub fn main() {
    assert!(("hello".to_string() < "hellr".to_string()));
    assert!(("hello ".to_string() > "hello".to_string()));
    assert!(("hello".to_string() != "there".to_string()));
    assert!((vec![1, 2, 3, 4] > vec![1, 2, 3]));
    assert!((vec![1, 2, 3] < vec![1, 2, 3, 4]));
    assert!((vec![1, 2, 4, 4] > vec![1, 2, 3, 4]));
    assert!((vec![1, 2, 3, 4] < vec![1, 2, 4, 4]));
    assert!((vec![1, 2, 3] <= vec![1, 2, 3]));
    assert!((vec![1, 2, 3] <= vec![1, 2, 3, 3]));
    assert!((vec![1, 2, 3, 4] > vec![1, 2, 3]));
    assert_eq!(vec![1, 2, 3], vec![1, 2, 3]);
    assert!((vec![1, 2, 3] != vec![1, 1, 3]));
}
