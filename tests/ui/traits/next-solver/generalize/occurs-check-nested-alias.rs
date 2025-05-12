//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// case 3 of https://github.com/rust-lang/trait-system-refactor-initiative/issues/8.
#![crate_type = "lib"]
#![allow(unused)]
trait Unnormalizable {
    type Assoc;
}

trait Id<T> {
    type Id;
}
impl<T, U> Id<U> for T {
    type Id = T;
}

struct Inv<T>(*mut T);

fn unconstrained<T>() -> T {
    todo!()
}

fn create<T: Unnormalizable, U>(
    x: &T,
) -> (Inv<U>, Inv<<<T as Id<U>>::Id as Unnormalizable>::Assoc>) {
    todo!()
}

fn foo<T: Unnormalizable>() {
    let t = unconstrained();
    let (mut u, assoc) = create::<_, _>(&t);
    u = assoc;
    // Instantiating `?u` with `<<?t as Id<?u>>::Id as Unnormalizable>::Assoc` would
    // result in a cyclic type. However, we can still unify these types by first
    // normalizing the inner associated type. Emitting an error here would be incomplete.
    drop::<T>(t);
}
