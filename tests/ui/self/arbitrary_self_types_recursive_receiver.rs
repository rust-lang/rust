//@ run-pass
#![feature(arbitrary_self_types)]

// When probing for methods, we step forward through a chain of types. The first
// few of those steps can be reached by jumping through the chain of Derefs or the
// chain of Receivers. Later steps can only be reached by following the chain of
// Receivers. For instance, supposing A and B implement both Receiver and Deref,
// while C and D implement only Receiver:
//
// Type A<B<C<D<E>>>>
//
//     Deref chain:      A -> B -> C
//     Receiver chain:   A -> B -> C -> D -> E
//
// We report bad type errors from the end of the chain. But at the end of which
// chain? We never morph the type as far as E so the correct behavior is to
// report errors from point C, i.e. the end of the Deref chain. This test case
// ensures we do that.

struct MyNonNull<T>(*const T);

impl<T> std::ops::Receiver for MyNonNull<T> {
    type Target = T;
}

#[allow(dead_code)]
impl<T> MyNonNull<T> {
    fn foo<U>(&self) -> *const U {
        let mnn = self.cast::<U>();
        // The following method call is the point of this test.
        // If probe.rs reported errors from the last type discovered
        // in the Receiver chain, it would be sad here because U is just
        // a type variable. But this is a valid call so it ensures
        // probe.rs doesn't make that mistake.
        mnn.bar()
    }
    fn cast<U>(&self) -> MyNonNull<U> {
        MyNonNull(self.0 as *const U)
    }
    fn bar(&self) -> *const T {
        self.0
    }
}

#[repr(transparent)]
struct Foo(usize);
#[repr(transparent)]
struct Bar(usize);

fn main() {
    let a = Foo(3);
    let ptr = MyNonNull(&a);
    let _bar_ptr: *const Bar = ptr.foo();
}
