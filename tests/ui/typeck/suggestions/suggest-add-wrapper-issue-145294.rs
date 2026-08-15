// Suppress the suggestion that adding a wrapper.
// When expected_ty and expr_ty are the same ADT,
// we prefer to compare their internal generic params,
// so when the current variant corresponds to an unresolved infer,
// the suggestion is rejected.
// e.g. `Ok(Some("hi"))` is type of `Result<Option<&str>, _>`,
// where `E` is still an unresolved inference variable.

fn foo() -> Result<Option<String>, ()> {
    todo!()
}

#[derive(PartialEq, Debug)]
enum Bar<T, E> {
    A(T),
    B(E),
}

fn bar() -> Bar<String, ()> {
    todo!()
}

fn main() {
    assert_eq!(Ok(Some("hi")), foo()); //~ ERROR mismatched types [E0308]
    assert_eq!(Bar::A("hi"), bar()); //~ ERROR mismatched types [E0308]
}
