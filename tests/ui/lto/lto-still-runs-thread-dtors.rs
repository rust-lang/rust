//@ run-pass
//@ compile-flags: -C lto
//@ no-prefer-dynamic
//@ needs-threads

use std::thread;

static mut HIT: usize = 0;

thread_local!(static A: Foo = Foo);

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        unsafe {
            HIT += 1;
        }
    }
}

fn main() {
    unsafe {
        assert_eq!(HIT, 0);
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        thread::spawn(|| {
            assert_eq!(HIT, 0);
            //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
            A.with(|_| ());
            assert_eq!(HIT, 0);
            //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        }).join().unwrap();
        assert_eq!(HIT, 1);
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
    }
}
