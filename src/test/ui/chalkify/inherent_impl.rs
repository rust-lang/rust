// run-pass
// compile-flags: -Z chalk
// FIXME(chalk): remove when uncommented
#![allow(dead_code, unused_variables)]

trait Foo { }

impl Foo for i32 { }

struct S<T: Foo> {
    x: T,
}

// FIXME(chalk): need late-bound regions on FnDefs
/*
fn only_foo<T: Foo>(_x: &T) { }

impl<T> S<T> {
    // Test that we have the correct environment inside an inherent method.
    fn dummy_foo(&self) {
        only_foo(&self.x)
    }
}
*/

trait Bar { }
impl Bar for u32 { }

fn only_bar<T: Bar>() { }

impl<T> S<T> {
    // Test that the environment of `dummy_bar` adds up with the environment
    // of the inherent impl.
    // FIXME(chalk): need late-bound regions on FnDefs
    /*
    fn dummy_bar<U: Bar>(&self) {
        only_foo(&self.x);
        only_bar::<U>();
    }
    */
    fn dummy_bar<U: Bar>() {
        only_bar::<U>();
    }
}

fn main() {
    let s = S {
        x: 5,
    };

    // FIXME(chalk): need late-bound regions on FnDefs
    /*
    s.dummy_foo();
    s.dummy_bar::<u32>();
    */
    S::<i32>::dummy_bar::<u32>();
}
