//@ revisions: old next
//@[old] check-pass

// Currently always fails to generalize the outer alias, even if it
// is treated as rigid by `alias-relate`.
//@[next] compile-flags: -Znext-solver
//@[next] known-bug: trait-system-refactor-initiative#8
#![crate_type = "lib"]
#![allow(unused)]
trait Unnormalizable {
    type Assoc;
}

trait Id<T> {
    type Id;
}
impl<T, U> Id<T> for U {
    type Id = U;
}

struct Inv<T>(*mut T);

fn unconstrained<T>() -> T {
    todo!()
}

fn create<T, U: Unnormalizable>(
    x: &U,
) -> (Inv<T>, Inv<<<U as Id<T>>::Id as Unnormalizable>::Assoc>) {
    todo!()
}

fn foo<T: Unnormalizable>() {
    let q = unconstrained();
    let (mut x, y) = create::<_, _>(&q);
    x = y;
    drop::<T>(q);
}
