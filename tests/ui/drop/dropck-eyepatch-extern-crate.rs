//@ run-pass
//@ aux-build:dropck_eyepatch_extern_crate.rs

extern crate dropck_eyepatch_extern_crate as other;

use other::{Dt,Dr,Pt,Pr,St,Sr};

fn main() {
    use std::cell::RefCell;

    struct CheckOnDrop(RefCell<String>, &'static str);
    impl Drop for CheckOnDrop {
        fn drop(&mut self) { assert_eq!(*self.0.borrow(), self.1); }
    }

    let c_long;
    let (c, dt, dr, pt, pr, st, sr)
        : (CheckOnDrop, Dt<_>, Dr<_>, Pt<_, _>, Pr<_>, St<_>, Sr<_>);
    c_long = CheckOnDrop(RefCell::new("c_long".to_string()),
                         "c_long|pr|pt|dr|dt");
    c = CheckOnDrop(RefCell::new("c".to_string()),
                    "c");

    // No error: sufficiently long-lived state can be referenced in dtors
    dt = Dt("dt", &c_long.0);
    dr = Dr("dr", &c_long.0);

    // No error: Drop impl asserts .1 (A and &'a _) are not accessed
    pt = Pt("pt", &c.0, &c_long.0);
    pr = Pr("pr", &c.0, &c_long.0);

    // No error: St and Sr have no destructor.
    st = St("st", &c.0);
    sr = Sr("sr", &c.0);

    println!("{:?}", (dt.0, dr.0, pt.0, pr.0, st.0, sr.0));
    assert_eq!(*c_long.0.borrow(), "c_long");
    assert_eq!(*c.0.borrow(), "c");
}
