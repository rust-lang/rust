//@ check-pass

use std::marker::PhantomData;
use std::mem;

trait Container<'a> {
    type Root: 'a;
}

type RootOf<'a, T> = <T as Container<'a>>::Root;

struct Test<'a, T> where T: Container<'a> {
    pub root: T::Root,
    marker: PhantomData<&'a mut &'a mut ()>,
}

impl<'a, 'b> Container<'b> for &'a str {
    type Root = &'b str;
}

impl<'a, T> Test<'a, T> where T: for<'b> Container<'b> {
    fn new(root: RootOf<'a, T>) -> Test<'a, T> {
        Test {
            root: root,
            marker: PhantomData
        }
    }

    fn with_mut<F, R>(&mut self, f: F) -> R where
            F: for<'b> FnOnce(&'b mut RootOf<'b, T>) -> R {
        f(unsafe { mem::transmute(&mut self.root) })
    }
}

fn main() {
    let val = "root";
    let mut test: Test<&str> = Test::new(val);
    test.with_mut(|_| { });
}
