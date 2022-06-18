use std::sync::mpsc::channel;
use std::thread::spawn;
use std::marker::PhantomData;

struct Foo<T> {foo: PhantomData<T>}

fn main() {
    let (tx, rx) = //~ ERROR type annotations needed
        channel();
    // FIXME(#89862): Suggest adding a generic argument to `channel` instead
    spawn(move || {
        tx.send(Foo{ foo: PhantomData });
    });
}
