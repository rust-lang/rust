// Test that `Clone` is correctly implemented for builtin types.
// Also test that cloning an array or a tuple is done right, i.e.
// each component is cloned.

fn test_clone<T: Clone>(arg: T) {
    let _ = arg.clone();
}

fn foo() {}

#[derive(Debug, PartialEq, Eq)]
struct S(i32);

impl Clone for S {
    fn clone(&self) -> Self {
        S(self.0 + 1)
    }
}

#[test]
fn builtin_clone() {
    test_clone(foo);
    test_clone([1; 56]);
    test_clone((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));

    let a = [S(0), S(1), S(2)];
    let b = [S(1), S(2), S(3)];
    assert_eq!(b, a.clone());

    let a = ((S(1), S(0)), ((S(0), S(0), S(1)), S(0)));
    let b = ((S(2), S(1)), ((S(1), S(1), S(2)), S(1)));
    assert_eq!(b, a.clone());
}
