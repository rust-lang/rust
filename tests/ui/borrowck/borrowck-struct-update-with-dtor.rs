// Issue 4691: Ensure that functional-struct-update can only copy, not
// move, when the struct implements Drop.

struct B;
struct S<K> { a: isize, b: B, c: K }
impl<K> Drop for S<K> { fn drop(&mut self) { } }

struct T { a: isize, b: Box<isize> }
impl Drop for T { fn drop(&mut self) { } }

struct V<K> { a: isize, b: Box<isize>, c: K }
impl<K> Drop for V<K> { fn drop(&mut self) { } }

#[derive(Clone)]
struct Clonable;

mod not_all_clone {
    use super::*;
    fn a(s0: S<()>) {
        let _s2 = S { a: 2, ..s0 };
        //~^ ERROR [E0509]
    }
    fn b(s0: S<B>) {
        let _s2 = S { a: 2, ..s0 };
        //~^ ERROR [E0509]
        //~| ERROR [E0509]
    }
    fn c<K: Clone>(s0: S<K>) {
        let _s2 = S { a: 2, ..s0 };
        //~^ ERROR [E0509]
        //~| ERROR [E0509]
    }
}
mod all_clone {
    use super::*;
    fn a(s0: T) {
        let _s2 = T { a: 2, ..s0 };
        //~^ ERROR [E0509]
    }

    fn b(s0: T) {
        let _s2 = T { ..s0 };
        //~^ ERROR [E0509]
    }

    fn c(s0: T) {
        let _s2 = T { a: 2, b: s0.b };
        //~^ ERROR [E0509]
    }

    fn d<K: Clone>(s0: V<K>) {
        let _s2 = V { a: 2, ..s0 };
        //~^ ERROR [E0509]
        //~| ERROR [E0509]
    }

    fn e(s0: V<Clonable>) {
        let _s2 = V { a: 2, ..s0 };
        //~^ ERROR [E0509]
        //~| ERROR [E0509]
    }
}

fn main() { }
