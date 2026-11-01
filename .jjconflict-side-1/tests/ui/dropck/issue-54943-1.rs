// This test is a minimal version of an ICE in the dropck-eyepatch tests
// found in the fix for #54943.

//@ check-pass

fn foo<T>(_t: T) {
}

fn main() {
    struct A<'a, B: 'a>(&'a B);
    let (a1, a2): (String, A<_>) = (String::from("auto"), A(&"this"));
    foo((a1, a2));
}
