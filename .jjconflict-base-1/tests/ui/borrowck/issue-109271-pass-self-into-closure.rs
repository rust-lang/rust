//@ run-rustfix
#![allow(unused)]
struct S;

impl S {
    fn call(&mut self, f: impl FnOnce((), &mut Self)) {
        // change state or something ...
        f((), self);
        // change state or something ...
    }

    fn get(&self) {}
    fn set(&mut self) {}
}

fn main() {
    let mut v = S;

    v.call(|(), this: &mut S| v.get());
    //~^ error: cannot borrow `v` as mutable because it is also borrowed as immutable
    v.call(|(), this: &mut S| v.set());
    //~^ error: cannot borrow `v` as mutable more than once at a time
    //~| error: cannot borrow `v` as mutable more than once at a time

    v.call(|(), this: &mut S| {
        //~^ error: cannot borrow `v` as mutable more than once at a time
        //~| error: cannot borrow `v` as mutable more than once at a time

        _ = v;
        v.set();
        v.get();
        S::get(&v);

        use std::ops::Add;
        let v = 0u32;
        _ = v + v;
        _ = v.add(3);
    });
}
