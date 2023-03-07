// run-pass
// Binding unsized expressions to ref patterns

pub fn main() {
    let ref a = *"abcdef";
    assert_eq!(a, "abcdef");

    match *"12345" {
        ref b => { assert_eq!(b, "12345") }
    }
}
