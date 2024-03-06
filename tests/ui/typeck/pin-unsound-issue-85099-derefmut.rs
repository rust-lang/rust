//@ check-pass
//@ known-bug: #85099

// Should fail. Can coerce `Pin<T>` into `Pin<U>` where
// `T: Deref<Target: Unpin>` and `U: Deref<Target: !Unpin>`, using the
// `CoerceUnsized` impl on `Pin` and an unorthodox `DerefMut` impl for
// `Pin<&_>`.

// This should not be allowed, since one can unpin `T::Target` (since it is
// `Unpin`) to gain unpinned access to the previously pinned `U::Target` (which
// is `!Unpin`) and then move it.

use std::{
    cell::{RefCell, RefMut},
    future::Future,
    ops::DerefMut,
    pin::Pin,
};

struct SomeLocalStruct<'a, Fut>(&'a RefCell<Fut>);

trait SomeTrait<'a, Fut> {
    #[allow(clippy::mut_from_ref)]
    fn deref_helper(&self) -> &mut (dyn SomeTrait<'a, Fut> + 'a) {
        unimplemented!()
    }
    fn downcast(self: Pin<&mut Self>) -> Pin<&mut Fut> {
        unimplemented!()
    }
}

impl<'a, Fut: Future<Output = ()>> SomeTrait<'a, Fut> for SomeLocalStruct<'a, Fut> {
    fn deref_helper(&self) -> &mut (dyn SomeTrait<'a, Fut> + 'a) {
        let x = Box::new(self.0.borrow_mut());
        let x: &'a mut RefMut<'a, Fut> = Box::leak(x);
        &mut **x
    }
}
impl<'a, Fut: Future<Output = ()>> SomeTrait<'a, Fut> for Fut {
    fn downcast(self: Pin<&mut Self>) -> Pin<&mut Fut> {
        self
    }
}

impl<'b, 'a, Fut> DerefMut for Pin<&'b dyn SomeTrait<'a, Fut>> {
    fn deref_mut<'c>(
        self: &'c mut Pin<&'b dyn SomeTrait<'a, Fut>>,
    ) -> &'c mut (dyn SomeTrait<'a, Fut> + 'b) {
        self.deref_helper()
    }
}

// obviously a "working" function with this signature is problematic
pub fn unsound_pin<Fut: Future<Output = ()>>(
    fut: Fut,
    callback: impl FnOnce(Pin<&mut Fut>),
) -> Fut {
    let cell = RefCell::new(fut);
    let s: &SomeLocalStruct<'_, Fut> = &SomeLocalStruct(&cell);
    let p: Pin<Pin<&SomeLocalStruct<'_, Fut>>> = Pin::new(Pin::new(s));
    let mut p: Pin<Pin<&dyn SomeTrait<'_, Fut>>> = p;
    let r: Pin<&mut dyn SomeTrait<'_, Fut>> = p.as_mut();
    let f: Pin<&mut Fut> = r.downcast();
    callback(f);
    cell.into_inner()
}

fn main() {}
