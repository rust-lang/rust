//@ run-pass
// Check that method matching does not make "guesses" depending on
// Deref impls that don't eventually end up being picked.

use std::ops::Deref;

// An impl with less derefs will get called over an impl with more derefs,
// so `(t: Foo<_>).my_fn()` will use `<Foo<u32> as MyTrait1>::my_fn(t)`,
// and does *not* force the `_` to equal `()`, because the Deref impl
// was *not* used.

trait MyTrait1 {
    fn my_fn(&self) {}
}

impl MyTrait1 for Foo<u32> {}

struct Foo<T>(#[allow(dead_code)] T);

impl Deref for Foo<()> {
    type Target = dyn MyTrait1 + 'static;
    fn deref(&self) -> &(dyn MyTrait1 + 'static) {
        panic!()
    }
}

// ...but if there is no impl with less derefs, the "guess" will be
// forced, so `(t: Bar<_>).my_fn2()` is `<dyn MyTrait2 as MyTrait2>::my_fn2(*t)`,
// and because the deref impl is used, the `_` is forced to equal `u8`.

trait MyTrait2 {
    fn my_fn2(&self) {}
}

impl MyTrait2 for u32 {}
struct Bar<T>(#[allow(dead_code)] T, u32);
impl Deref for Bar<u8> {
    type Target = dyn MyTrait2 + 'static;
    fn deref(&self) -> &(dyn MyTrait2 + 'static) {
        &self.1
    }
}

// actually invoke things

fn main() {
    let mut foo: Option<Foo<_>> = None;
    let mut bar: Option<Bar<_>> = None;
    let mut first_iter = true;
    loop {
        if !first_iter {
            foo.as_ref().unwrap().my_fn();
            bar.as_ref().unwrap().my_fn2();
            break;
        }
        foo = Some(Foo(0));
        bar = Some(Bar(Default::default(), 0));
        first_iter = false;
    }
}
