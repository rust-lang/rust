// Test for <https://github.com/rust-lang/rust/issues/135315>.
//
//@ run-pass
//@ check-run-results

trait Supertrait<T> {
    fn _print_numbers(&self, mem: &[usize; 100]) {
        println!("{mem:?}");
    }
}
impl<T> Supertrait<T> for () {}

trait Identity {
    type Selff;
}
impl<Selff> Identity for Selff {
    type Selff = Selff;
}

trait Middle<T>: Supertrait<()> + Supertrait<T> {
    fn say_hello(&self, _: &usize) {
        println!("Hello!");
    }
}
impl<T> Middle<T> for () {}

trait Trait: Middle<<() as Identity>::Selff> {}
impl Trait for () {}

fn main() {
    (&() as &dyn Trait as &dyn Middle<()>).say_hello(&0);
}
