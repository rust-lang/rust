// run-pass
// Test that for loops can do what RFC #235 claims


fn main() {
    let mut v = vec![1];

    for x in &v {
        assert_eq!(x, &1);
    }

    for x in &mut v {
        assert_eq!(x, &mut 1);
    }

    for x in v {
        assert_eq!(x, 1);
    }
}
