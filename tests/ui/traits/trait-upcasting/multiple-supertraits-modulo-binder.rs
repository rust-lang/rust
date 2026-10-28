// Test for <https://github.com/rust-lang/rust/issues/135316>.
//
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[current] run-pass
//@[current] check-run-results

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
    (&() as &'static dyn for<'a> Trait<&'static (), &'a ()>
        as &'static dyn Trait<&'static (), &'static ()>)
        .say_hello(&0); //[next]~ ERROR: type annotations needed
}
