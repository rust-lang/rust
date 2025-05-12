//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass
#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

trait WithAssoc: 'static {
    type Assoc;
}
impl<T: 'static> WithAssoc for (T,) {
    type Assoc = ();
}

struct GenericArray<U: WithAssoc>(U::Assoc);

trait AbiExample {
    fn example();
}
impl<U: WithAssoc> AbiExample for GenericArray<U> {
    fn example() {}
}
impl<T> AbiExample for T {
    default fn example() {}
}

fn main() {
    let _ = GenericArray::<((),)>::example();
}
