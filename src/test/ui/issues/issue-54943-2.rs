// This test is a minimal version of an ICE in the dropck-eyepatch tests
// found in the fix for #54943. In particular, this test is in unreachable
// code as the initial fix for this ICE only worked if the code was reachable.

// build-pass (FIXME(62277): could be check-pass?)

fn foo<T>(_t: T) {
}

fn main() {
    return;

    struct A<'a, B: 'a>(&'a B);
    let (a1, a2): (String, A<_>) = (String::from("auto"), A(&"this"));
    foo((a1, a2));
}
