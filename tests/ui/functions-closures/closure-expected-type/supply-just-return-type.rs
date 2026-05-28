//@ run-pass
fn with_closure<F, R>(f: F) -> Result<char, R>
    where F: FnOnce(&char) -> Result<char, R>,
{
    f(&'a')
}

fn main() {
    // Test that supplying the `-> Result<char, ()>` manually here
    // (which is needed to constrain `R`) still allows us to figure
    // out that the type of `x` is `&'a char` where `'a` is bound in
    // the closure (if we didn't, we'd get a type-error because
    // `with_closure` requires a bound region).
    //
    // This pattern was found in the wild.
    let z = with_closure(|x| -> Result<char, ()> { Ok(*x) });
    assert_eq!(z.unwrap(), 'a');

    // It also works with `_`:
    let z = with_closure(|x: _| -> Result<char, ()> { Ok(*x) });
    assert_eq!(z.unwrap(), 'a');

    // It also works with `&_`:
    let z = with_closure(|x: &_| -> Result<char, ()> { Ok(*x) });
    assert_eq!(z.unwrap(), 'a');
}
