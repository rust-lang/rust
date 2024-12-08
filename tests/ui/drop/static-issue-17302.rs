//@ run-pass

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

static mut DROPPED: [bool; 2] = [false, false];

struct A(usize);
struct Foo { _a: A, _b: isize }

impl Drop for A {
    fn drop(&mut self) {
        let A(i) = *self;
        unsafe { DROPPED[i] = true; }
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
        assert!(DROPPED[1]);
    }
}
