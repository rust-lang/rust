//! Test that dead code warnings are issued for superfluous assignments of fields or variables to
//! themselves (issue #75356).
//!
//! # History of this test (to aid relanding of a fixed version of #81473)
//!
//! - Original lint request was about self-assignments not triggering sth like `dead_code`.
//! - `dead_code` lint expansion for self-assignments was implemented in #87129.
//! - Unfortunately implementation components of #87129 had to be disabled as part of reverts
//!   #86212, #83171 (to revert #81473) to address regressions #81626 and #81658.
//! - Consequently, none of the following warnings are emitted.

//@ check-pass

// Implementation of self-assignment `dead_code` lint expansions disabled due to reverts.
//@ known-bug: #75356

#![allow(unused_assignments)]
#![warn(dead_code)]

fn main() {
    let mut x = 0;
    x = x;
    // FIXME ~^ WARNING: useless assignment of variable of type `i32` to itself

    x = (x);
    // FIXME ~^ WARNING: useless assignment of variable of type `i32` to itself

    x = {x};
    // block expressions don't count as self-assignments


    struct S<'a> { f: &'a str }
    let mut s = S { f: "abc" };
    s = s;
    // FIXME ~^ WARNING: useless assignment of variable of type `S` to itself

    s.f = s.f;
    // FIXME ~^ WARNING: useless assignment of field of type `&str` to itself


    struct N0 { x: Box<i32> }
    struct N1 { n: N0 }
    struct N2(N1);
    struct N3 { n: N2 };
    let mut n3 = N3 { n: N2(N1 { n: N0 { x: Box::new(42) } }) };
    n3.n.0.n.x = n3.n.0.n.x;
    // FIXME ~^ WARNING: useless assignment of field of type `Box<i32>` to itself

    let mut t = (1, ((2, 3, (4, 5)),));
    t.1.0.2.1 = t.1.0.2.1;
    // FIXME ~^ WARNING: useless assignment of field of type `i32` to itself


    let mut y = 0;
    macro_rules! assign_to_y {
        ($cur:expr) => {{
            y = $cur;
        }};
    }
    assign_to_y!(y);
    // self-assignments in macro expansions are not reported either
}
