// Ensure that arguments are properly handled when providing suggestion for alternative in method
// chains. Regression test for #166275.

use std::sync::atomic::{AtomicU8, Ordering};

struct List<'a> {
    a: Option<&'a i32>,
}

struct Foo<T>(T);
struct Bar;
trait Trait<T> {}

impl Foo<u8> {
    fn make_bar(self) -> Bar {
        Bar
    }
}

impl<T> Foo<Option<T>> {
    fn make_bar(self, _: impl Trait<T>) -> Bar {
        Bar
    }
}

fn main() {
    let _ = AtomicU8::new(0).try_update(Ordering::AcqRel, Ordering::Acquire, |old| Some(2)).load();
    //~^ ERROR
    Foo(1_u8).make_bar().other(); //~ ERROR
    List { a: Some(&3) }.a.cloned().kaboom; //~ ERROR
}
