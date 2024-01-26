//@ run-pass
#![feature(dropck_eyepatch)]
#![allow(non_local_definitions)]

// The point of this test is to illustrate that the `#[may_dangle]`
// attribute specifically allows, in the context of a type
// implementing `Drop`, a generic parameter to be instantiated with a
// lifetime that does not strictly outlive the owning type itself.
//
// Here we test that a model use of `#[may_dangle]` will compile and run.
//
// The illustration is made concrete by comparison with two variations
// on the type with `#[may_dangle]`:
//
//   1. an analogous type that does not implement `Drop` (and thus
//      should exhibit maximal flexibility with respect to dropck), and
//
//   2. an analogous type that does not use `#[may_dangle]` (and thus
//      should exhibit the standard limitations imposed by dropck.
//
// The types in this file follow a pattern, {D,P,S}{t,r}, where:
//
// - D means "I implement Drop"
//
// - P means "I implement Drop but guarantee my (first) parameter is
//     pure, i.e., not accessed from the destructor"; no other parameters
//     are pure.
//
// - S means "I do not implement Drop"
//
// - t suffix is used when the first generic is a type
//
// - r suffix is used when the first generic is a lifetime.

trait Foo { fn foo(&self, _: &str); }

struct Dt<A: Foo>(&'static str, A);
struct Dr<'a, B:'a+Foo>(&'static str, &'a B);
struct Pt<A,B: Foo>(&'static str, #[allow(dead_code)] A, B);
struct Pr<'a, 'b, B:'a+'b+Foo>(&'static str, #[allow(dead_code)] &'a B, &'b B);
struct St<A: Foo>(&'static str, #[allow(dead_code)] A);
struct Sr<'a, B:'a+Foo>(&'static str, #[allow(dead_code)] &'a B);

impl<A: Foo> Drop for Dt<A> {
    fn drop(&mut self) { println!("drop {}", self.0); self.1.foo(self.0); }
}
impl<'a, B: Foo> Drop for Dr<'a, B> {
    fn drop(&mut self) { println!("drop {}", self.0); self.1.foo(self.0); }
}
unsafe impl<#[may_dangle] A, B: Foo> Drop for Pt<A, B> {
    // (unsafe to access self.1  due to #[may_dangle] on A)
    fn drop(&mut self) { println!("drop {}", self.0); self.2.foo(self.0); }
}
unsafe impl<#[may_dangle] 'a, 'b, B: Foo> Drop for Pr<'a, 'b, B> {
    // (unsafe to access self.1 due to #[may_dangle] on 'a)
    fn drop(&mut self) { println!("drop {}", self.0); self.2.foo(self.0); }
}

fn main() {
    use std::cell::RefCell;

    impl Foo for RefCell<String> {
        fn foo(&self, s: &str) {
            let s2 = format!("{}|{}", *self.borrow(), s);
            *self.borrow_mut() = s2;
        }
    }

    impl<'a, T:Foo> Foo for &'a T {
        fn foo(&self, s: &str) {
            (*self).foo(s);
        }
    }

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
