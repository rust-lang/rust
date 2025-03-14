// Test for <https://github.com/rust-lang/rust/issues/135316>.
//
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
    (&() as &'static dyn for<'a> Trait<&'static (), &'a ()>
        as &'static dyn Trait<&'static (), &'static ()>)
        .say_hello(&0);
}
