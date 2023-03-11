// ignore-wasm32
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck
// normalize-stderr-test: "__LocalKeyInner::<T>::get" -> "$$LOCALKEYINNER::<T>::get"
// normalize-stderr-test: "__LocalKeyInner::<T>::get" -> "$$LOCALKEYINNER::<T>::get"
#![feature(thread_local)]
#![feature(cfg_target_thread_local, thread_local_internals)]

use std::cell::RefCell;

type Foo = std::cell::RefCell<String>;

#[cfg(target_thread_local)]
#[thread_local]
static __KEY: std::thread::__LocalKeyInner<Foo> = std::thread::__LocalKeyInner::new();

#[cfg(not(target_thread_local))]
static __KEY: std::thread::__LocalKeyInner<Foo> = std::thread::__LocalKeyInner::new();

fn __getit(_: Option<&mut Option<RefCell<String>>>) -> std::option::Option<&'static Foo> {
    __KEY.get(Default::default)
    //[mir]~^ ERROR call to unsafe function is unsafe
    //[thir]~^^ ERROR call to unsafe function `__
}

static FOO: std::thread::LocalKey<Foo> = std::thread::LocalKey::new(__getit);
//[mir]~^ ERROR call to unsafe function is unsafe
//[thir]~^^ ERROR call to unsafe function `LocalKey::<T>::new`

fn main() {
    FOO.with(|foo| println!("{}", foo.borrow()));
    std::thread::spawn(|| {
        FOO.with(|foo| *foo.borrow_mut() += "foo");
    })
    .join()
    .unwrap();
    FOO.with(|foo| println!("{}", foo.borrow()));
}
