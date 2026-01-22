// The same as check-wf-of-closure-args-complex, but this example causes
// a use-after-free if we don't perform the wf check.
//
#![feature(unboxed_closures)]

use std::sync::OnceLock;

type Payload = Box<i32>;

static STORAGE: OnceLock<&'static Payload> = OnceLock::new();

trait Store {
    fn store(&self);
}
impl Store for &'static Payload {
    fn store(&self) {
        STORAGE.set(*self).unwrap();
    }
}

#[repr(transparent)]
struct MyTy<T: Store>(T);
impl<T: Store> Drop for MyTy<T> {
    fn drop(&mut self) {
        self.0.store();
    }
}

trait IsFn: for<'x> Fn<(&'x Payload,)> {}

impl IsFn for for<'x> fn(&'x Payload) -> MyTy<&'x Payload> {}

fn foo(f: impl for<'x> Fn(&'x Payload)) {
    let a = Box::new(1);
    f(&a);
}
fn bar<F: IsFn>(f: F) {
    foo(|x| { f(x); });
}

fn main() {
    // If no wf-check is done on this closure given to `bar`, this compiles fine and
    // a use-after-free will occur.
    bar::<for<'a> fn(&'a Payload) -> MyTy<&'a Payload>>(|x| unsafe { //~ ERROR: lifetime may not live long enough
        std::mem::transmute::<&Payload, MyTy<&Payload>>(x)
    });
    println!("{}", STORAGE.get().unwrap());
}
