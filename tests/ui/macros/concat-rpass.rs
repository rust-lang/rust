//@ run-pass

pub fn main() {
    assert_eq!(format!(concat!("foo", "bar", "{}"), "baz"), "foobarbaz".to_string());
    assert_eq!(format!(concat!()), "".to_string());
    // check trailing comma is allowed in concat
    assert_eq!(concat!("qux", "quux",).to_string(), "quxquux".to_string());

    assert_eq!(
        concat!(1, 2, 3, 4f32, 4.0, 'a', true),
        "12344.0atrue"
    );

    assert!(match "12344.0atrue" {
        concat!(1, 2, 3, 4f32, 4.0, 'a', true) => true,
        _ => false
    })
}
