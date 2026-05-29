//@ check-pass
//@ compile-flags: -Z validate-mir
//@ edition: 2024
#![allow(todo_macro_calls)]

fn lambda<T, U>() -> U
where
    T: Default,
    U: Default,
{
    let foo: Result<T, ()> = Ok(T::default());
    let baz: U = U::default();

    if let Ok(foo) = foo && let Ok(bar) = transform(foo) {
        bar
    } else {
        baz
    }
}

fn transform<T, U>(input: T) -> Result<U, ()> {
    todo!()
}

fn main() {}
