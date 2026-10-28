// Test for <https://github.com/rust-lang/rust/issues/135316>.
//
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ run-pass
//@ check-run-results

trait Supertrait<T> {
    fn _print_numbers(&self, mem: &[usize; 100]) {
        println!("{mem:?}");
    }
}
impl<T> Supertrait<T> for () {}

trait Trait<T, U>: Supertrait<T> + Supertrait<U> {
    fn say_hello(&self, _: &usize) {
        println!("Hello!");
    }
}
impl<T, U> Trait<T, U> for () {}

fn main() {
    // We can't have `dyn Trait<&'static (), &'static ()>` directly here
    // as that results in an ambiguity error in MIR borrowck.
    fn generic<T: 'static>() {
        (&() as &'static dyn for<'a> Trait<T, &'a ()> as &'static dyn Trait<T, &'static ()>)
            .say_hello(&0);
    }

    generic::<&'static ()>();
}
