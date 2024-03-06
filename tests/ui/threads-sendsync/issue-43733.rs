//@ needs-threads
#![feature(thread_local)]
#![feature(cfg_target_thread_local, thread_local_internals)]

use std::cell::RefCell;

type Foo = std::cell::RefCell<String>;

#[cfg(target_thread_local)]
#[thread_local]
static __KEY: std::thread::local_impl::Key<Foo> = std::thread::local_impl::Key::new();

#[cfg(not(target_thread_local))]
static __KEY: std::thread::local_impl::Key<Foo> = std::thread::local_impl::Key::new();

fn __getit(_: Option<&mut Option<RefCell<String>>>) -> std::option::Option<&'static Foo> {
    __KEY.get(Default::default)
    //~^ ERROR call to unsafe function `Key::<T>::get` is unsafe
}

static FOO: std::thread::LocalKey<Foo> = std::thread::LocalKey::new(__getit);
//~^ ERROR call to unsafe function `LocalKey::<T>::new` is unsafe

fn main() {
    FOO.with(|foo| println!("{}", foo.borrow()));
    std::thread::spawn(|| {
        FOO.with(|foo| *foo.borrow_mut() += "foo");
    })
    .join()
    .unwrap();
    FOO.with(|foo| println!("{}", foo.borrow()));
}
