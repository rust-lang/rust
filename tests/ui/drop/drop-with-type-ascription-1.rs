// run-pass

fn main() {
    let foo = "hello".to_string();
    let foo: Vec<&str> = foo.split_whitespace().collect();
    let invalid_string = &foo[0];
    assert_eq!(*invalid_string, "hello");
}
