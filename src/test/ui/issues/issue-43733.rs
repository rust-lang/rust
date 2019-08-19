#![feature(const_fn)]
#![feature(thread_local)]
#![feature(cfg_target_thread_local, thread_local_internals)]

type Foo = std::cell::RefCell<String>;

#[cfg(target_thread_local)]
#[thread_local]
static __KEY: std::thread::__FastLocalKeyInner<Foo> =
    std::thread::__FastLocalKeyInner::new();

#[cfg(not(target_thread_local))]
static __KEY: std::thread::__OsLocalKeyInner<Foo> =
    std::thread::__OsLocalKeyInner::new();

fn __getit() -> std::option::Option<&'static Foo>
{
    __KEY.get(Default::default) //~ ERROR call to unsafe function is unsafe
}

static FOO: std::thread::LocalKey<Foo> =
    std::thread::LocalKey::new(__getit);
//~^ ERROR call to unsafe function is unsafe

fn main() {
    FOO.with(|foo| println!("{}", foo.borrow()));
    std::thread::spawn(|| {
        FOO.with(|foo| *foo.borrow_mut() += "foo");
    }).join().unwrap();
    FOO.with(|foo| println!("{}", foo.borrow()));
}
