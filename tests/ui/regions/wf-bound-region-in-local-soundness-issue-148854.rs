// A regression test for https://github.com/rust-lang/rust/issues/148854.

use std::cell::OnceCell;
use std::fmt::Display;
use std::marker::PhantomData;
use std::rc::Rc;

type Storage = Rc<OnceCell<Box<dyn Display + 'static>>>;

trait IntoDyn<T> {
    fn into_dyn(input: T, output: Storage);
}

struct Inner<T: Display + 'static>(PhantomData<T>);
impl<T: Display> IntoDyn<T> for Inner<T> {
    fn into_dyn(input: T, output: Storage) {
        output.set(Box::new(input)).ok().unwrap();
    }
}

struct Outer<T, U: IntoDyn<T>> {
    input: Option<T>,
    output: Storage,
    _phantom: PhantomData<U>,
}
impl<T, U: IntoDyn<T>> Drop for Outer<T, U> {
    fn drop(&mut self) {
        U::into_dyn(self.input.take().unwrap(), self.output.clone());
    }
}

fn extend<T: Display>(x: T) -> Box<dyn Display + 'static> {
    let storage = Rc::new(OnceCell::new());
    {
        // This has to error due to an unsatisfied outlives bound on
        // `Inner<T: 'static>` as its implicit drop relies on that
        // bound.
        let _ =
            Outer::<T, Inner<T>> { input: Some(x), output: storage.clone(), _phantom: PhantomData };
        //~^ ERROR: the parameter type `T` may not live long enough
    }
    Rc::into_inner(storage).unwrap().into_inner().unwrap()
}

fn main() {
    let wrong = {
        let data = String::from("abc");
        extend::<&String>(&data)
    };
    println!("{wrong}");
}
