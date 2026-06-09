//! Tests cleanup behavior of the built-in `Clone` impl for tuples during unwinding.

//@ run-pass
//@ needs-unwind
//@ ignore-backends: gcc

#![allow(unused_variables)]
#![allow(unused_imports)]

use std::rc::Rc;
use std::thread;

struct S(Rc<()>);

impl Clone for S {
    fn clone(&self) -> Self {
        if Rc::strong_count(&self.0) == 7 {
            panic!("oops");
        }

        S(self.0.clone())
    }
}

fn main() {
    let counter = Rc::new(());

    // Unwinding with tuples...
    let ccounter = counter.clone();
    let result = std::panic::catch_unwind(move || {
        let _ =
            (S(ccounter.clone()), S(ccounter.clone()), S(ccounter.clone()), S(ccounter)).clone();
    });

    assert!(result.is_err());
    assert_eq!(1, Rc::strong_count(&counter));

    // ... and with arrays.
    let ccounter = counter.clone();
    let child = std::panic::catch_unwind(move || {
        let _ =
            [S(ccounter.clone()), S(ccounter.clone()), S(ccounter.clone()), S(ccounter)].clone();
    });

    assert!(child.is_err());
    assert_eq!(1, Rc::strong_count(&counter));
}
