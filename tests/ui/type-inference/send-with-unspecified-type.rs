use std::sync::mpsc::channel;
use std::thread::spawn;
use std::marker::PhantomData;

struct Foo<T> {foo: PhantomData<T>}

fn main() {
    let (tx, rx) =
        channel();
    spawn(move || {
        tx.send(Foo{ foo: PhantomData });
        //~^ ERROR type annotations needed
    });
}
