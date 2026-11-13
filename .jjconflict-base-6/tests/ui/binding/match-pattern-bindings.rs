//@ run-pass

fn main() {
    let value = Some(1);
    assert_eq!(match value {
        ref a @ Some(_) => a,
        ref b @ None => b
    }, &Some(1));
    assert_eq!(match value {
        ref c @ Some(_) => c,
        ref b @ None => b
    }, &Some(1));
    assert_eq!(match "foobarbaz" {
        b @ _ => b
    }, "foobarbaz");
    let a @ _ = "foobarbaz";
    assert_eq!(a, "foobarbaz");
    let value = Some(true);
    let ref a @ _ = value;
    assert_eq!(a, &Some(true));
}
