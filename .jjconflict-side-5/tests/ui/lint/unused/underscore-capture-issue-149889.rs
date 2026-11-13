//@ check-pass
#![deny(unused_assignments)]

fn lock() -> impl Drop {
    struct Handle;

    impl Drop for Handle {
        fn drop(&mut self) {}
    }

    Handle
}

fn bar(_f: impl FnMut(bool)) {}

pub fn foo() {
    let mut _handle = None;
    bar(move |l| {
        if l {
            _handle = Some(lock());
        } else {
            _handle = None;
        }
    })
}

fn main() {
    foo();
}
