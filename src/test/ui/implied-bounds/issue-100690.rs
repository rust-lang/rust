// This code (probably) _should_ compile, but it currently does not because we
// are not smart enough about implied bounds.

use std::io;

fn real_dispatch<T, F>(f: F) -> Result<(), io::Error>
//~^ NOTE required by a bound in this
where
    F: FnOnce(&mut UIView<T>) -> Result<(), io::Error> + Send + 'static,
    //~^ NOTE required by this bound in `real_dispatch`
    //~| NOTE required by a bound in `real_dispatch`
{
    todo!()
}

#[derive(Debug)]
struct UIView<'a, T: 'a> {
    _phantom: std::marker::PhantomData<&'a mut T>,
}

trait Handle<'a, T: 'a, V, R> {
    fn dispatch<F>(&self, f: F) -> Result<(), io::Error>
    where
        F: FnOnce(&mut V) -> R + Send + 'static;
}

#[derive(Debug, Clone)]
struct TUIHandle<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: 'a> Handle<'a, T, UIView<'a, T>, Result<(), io::Error>> for TUIHandle<T> {
    fn dispatch<F>(&self, f: F) -> Result<(), io::Error>
    where
        F: FnOnce(&mut UIView<'a, T>) -> Result<(), io::Error> + Send + 'static,
    {
        real_dispatch(f)
        //~^ ERROR expected a `FnOnce<(&mut UIView<'_, T>,)>` closure, found `F`
        //~| NOTE expected an `FnOnce<(&mut UIView<'_, T>,)>` closure, found `F`
        //~| NOTE expected a closure with arguments
        //~| NOTE required by a bound introduced by this call
    }
}

fn main() {}
