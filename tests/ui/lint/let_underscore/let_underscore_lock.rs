//@ check-fail
use std::sync::{Arc, Mutex};

struct Struct<T> {
    a: T,
}

fn identity<T>(t: T) -> T {
    t
}

fn main() {
    let data = Arc::new(Mutex::new(0));
    let _ = data.lock().unwrap(); //~ERROR non-binding let on a synchronization lock

    let _ = data.lock(); //~ERROR non-binding let on a synchronization lock

    let (_, _) = (data.lock(), 1); //~ERROR non-binding let on a synchronization lock

    let (_a, Struct { a: _ }) = (0, Struct { a: data.lock() }); //~ERROR non-binding let on a synchronization lock

    (_ , _) = (data.lock(), 1); //~ERROR non-binding let on a synchronization lock

    let _ = (data.lock(), 1).0; //~ERROR non-binding let on a synchronization lock

    let _ = *Box::new(data.lock()); //~ERROR non-binding let on a synchronization lock

    let _ = Box::new(Struct { a: data.lock() }).a; //~ERROR non-binding let on a synchronization lock

    let &_ = identity(&data.lock()); //~ERROR non-binding let on a synchronization lock

    let &_ = &*identity(&data.lock()); //~ERROR non-binding let on a synchronization lock

    let &&_ = &identity(&data.lock()); //~ERROR non-binding let on a synchronization lock

    let &&_ = &(identity(&data.lock()), 1).0; //~ERROR non-binding let on a synchronization lock

    let foo = Struct { a: data.lock() };
    let (_c, Struct { a: _ }) = (0, foo); //~ERROR non-binding let on a synchronization lock

    let _b;
    (_b, Struct { a: _ }) = (0, Struct { a: data.lock() }); //~ERROR non-binding let on a synchronization lock
}
