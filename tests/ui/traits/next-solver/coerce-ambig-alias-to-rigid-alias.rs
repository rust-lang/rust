//@ compile-flags: -Znext-solver
//@ check-pass

trait Trait {
    type Assoc;
}

fn call<T: Trait>(_: <T as Trait>::Assoc, _: T) {}

fn foo<T: Trait>(rigid: <T as Trait>::Assoc, t: T) {
    // Check that we can coerce `<?0 as Trait>::Assoc` to `<T as Trait>::Assoc`.
    call::<_ /* ?0 */>(rigid, t);
}

fn main() {}
