//@ run-pass

static mut DROPPED: [bool; 2] = [false, false];

struct A(usize);
struct Foo { _a: A, _b: isize }

impl Drop for A {
    fn drop(&mut self) {
        let A(i) = *self;
        unsafe { DROPPED[i] = true; }
        //~^ WARN creating a reference to mutable static is discouraged [static_mut_refs]
    }
}

fn main() {
    {
        Foo {
            _a: A(0),
            ..Foo { _a: A(1), _b: 2 }
        };
    }
    unsafe {
        assert!(DROPPED[0]);
        //~^ WARN creating a reference to mutable static is discouraged [static_mut_refs]
        assert!(DROPPED[1]);
        //~^ WARN creating a reference to mutable static is discouraged [static_mut_refs]
    }
}
