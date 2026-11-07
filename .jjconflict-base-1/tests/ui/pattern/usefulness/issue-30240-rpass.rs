//@ run-pass
fn main() {
    let &ref a = &[0i32] as &[_];
    assert_eq!(a, &[0i32] as &[_]);

    let &ref a = "hello";
    assert_eq!(a, "hello");

    match "foo" {
        "fool" => unreachable!(),
        "foo" => {},
        ref _x => unreachable!()
    }
}
