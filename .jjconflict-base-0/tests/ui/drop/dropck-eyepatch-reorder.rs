//@ run-pass
#![feature(dropck_eyepatch)]
#![allow(non_local_definitions)]

// The point of this test is to test uses of `#[may_dangle]` attribute
// where the formal declaration order (in the impl generics) does not
// match the actual usage order (in the type instantiation).
//
// See also dropck-eyepatch.rs for more information about the general
// structure of the test.

trait Foo { fn foo(&self, _: &str); }

struct Dt<A: Foo>(&'static str, A);
struct Dr<'a, B:'a+Foo>(&'static str, &'a B);
struct Pt<A: Foo, B: Foo>(&'static str, #[allow(dead_code)] A, B);
struct Pr<'a, 'b, B:'a+'b+Foo>(&'static str, #[allow(dead_code)] &'a B, &'b B);
struct St<A: Foo>(&'static str, #[allow(dead_code)] A);
struct Sr<'a, B:'a+Foo>(&'static str, #[allow(dead_code)] &'a B);

impl<A: Foo> Drop for Dt<A> {
    fn drop(&mut self) { println!("drop {}", self.0); self.1.foo(self.0); }
}
impl<'a, B: Foo> Drop for Dr<'a, B> {
    fn drop(&mut self) { println!("drop {}", self.0); self.1.foo(self.0); }
}
unsafe impl<B: Foo, #[may_dangle] A: Foo> Drop for Pt<A, B> {
    // (unsafe to access self.1  due to #[may_dangle] on A)
    fn drop(&mut self) { println!("drop {}", self.0); self.2.foo(self.0); }
}
unsafe impl<'b, #[may_dangle] 'a, B: Foo> Drop for Pr<'a, 'b, B> {
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
