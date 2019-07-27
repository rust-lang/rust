// run-pass
fn r#fn(r#match: u32) -> u32 {
    r#match
}

pub fn main() {
    let r#struct = 1;
    assert_eq!(1, r#struct);

    let foo = 2;
    assert_eq!(2, r#foo);

    let r#bar = 3;
    assert_eq!(3, bar);

    assert_eq!(4, r#fn(4));

    let r#true = false;
    assert_eq!(r#true, false);
}
