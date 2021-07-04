use std::marker::PhantomData;

fn main() {
    let error = Closure::wrap(Box::new(move || {
    //~^ ERROR type mismatch
        panic!("Can't connect to server.");
    }) as Box<dyn FnMut()>);
}

struct Closure<T: ?Sized>(PhantomData<T>);

impl<T: ?Sized> Closure<T> {
    fn wrap(data: Box<T>) -> Closure<T> {
        todo!()
    }
}
