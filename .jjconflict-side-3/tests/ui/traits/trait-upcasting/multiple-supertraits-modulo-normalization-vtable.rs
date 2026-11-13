#![feature(rustc_attrs)]

// Test for <https://github.com/rust-lang/rust/issues/135315>.

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

#[rustc_dump_vtable]
impl Trait for () {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
type Virtual = dyn Middle<()>;
//~^ ERROR vtable entries

fn main() {}
