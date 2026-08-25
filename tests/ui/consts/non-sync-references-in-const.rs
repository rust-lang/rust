//@ check-pass
//@ known-bug: #49206

// Should fail. Compiles and prints 2 identical addresses, which shows 2 threads
// with the same `'static` reference to non-`Sync` struct. The problem is that
// promotion to static does not check if the type is `Sync`.

#[allow(dead_code)]
#[derive(Debug)]
struct Foo {
    value: u32,
}

// stable negative impl trick from https://crates.io/crates/negative-impl
// see https://github.com/taiki-e/pin-project/issues/102#issuecomment-540472282
// for details.
struct Wrapper<'a, T>(::std::marker::PhantomData<&'a ()>, T);
unsafe impl<T> Sync for Wrapper<'_, T> where T: Sync {}
unsafe impl<'a> std::marker::Sync for Foo where Wrapper<'a, *const ()>: Sync {}
fn _assert_sync<T: Sync>() {}

fn inspect() {
    let foo: &'static Foo = &Foo { value: 1 };
    println!(
        "I am in thread {:?}, address: {:p}",
        std::thread::current().id(),
        foo as *const Foo,
    );
}

fn main() {
    // _assert_sync::<Foo>(); // uncomment this line causes compile error
    // "`*const ()` cannot be shared between threads safely"

    let handle = std::thread::spawn(inspect);
    inspect();
    handle.join().unwrap();
}
