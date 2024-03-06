//@ check-pass

use std::rc::Rc;
use std::sync::Arc;
use std::cmp::PartialEq;

struct A;
struct B;

trait T {}
impl T for A {}
impl T for B {}

fn main() {
    let ab = (A, B);
    let a = &ab.0 as *const dyn T;
    let b = &ab.1 as *const dyn T;

    let _ = a == b;
    //~^ WARN ambiguous wide pointer comparison
    let _ = a != b;
    //~^ WARN ambiguous wide pointer comparison
    let _ = a < b;
    //~^ WARN ambiguous wide pointer comparison
    let _ = a <= b;
    //~^ WARN ambiguous wide pointer comparison
    let _ = a > b;
    //~^ WARN ambiguous wide pointer comparison
    let _ = a >= b;
    //~^ WARN ambiguous wide pointer comparison

    let _ = PartialEq::eq(&a, &b);
    //~^ WARN ambiguous wide pointer comparison
    let _ = PartialEq::ne(&a, &b);
    //~^ WARN ambiguous wide pointer comparison
    let _ = a.eq(&b);
    //~^ WARN ambiguous wide pointer comparison
    let _ = a.ne(&b);
    //~^ WARN ambiguous wide pointer comparison

    {
        // &*const ?Sized
        let a = &a;
        let b = &b;

        let _ = a == b;
        //~^ WARN ambiguous wide pointer comparison
        let _ = a != b;
        //~^ WARN ambiguous wide pointer comparison
        let _ = a < b;
        //~^ WARN ambiguous wide pointer comparison
        let _ = a <= b;
        //~^ WARN ambiguous wide pointer comparison
        let _ = a > b;
        //~^ WARN ambiguous wide pointer comparison
        let _ = a >= b;
        //~^ WARN ambiguous wide pointer comparison

        let _ = PartialEq::eq(a, b);
        //~^ WARN ambiguous wide pointer comparison
        let _ = PartialEq::ne(a, b);
        //~^ WARN ambiguous wide pointer comparison
        let _ = PartialEq::eq(&a, &b);
        //~^ WARN ambiguous wide pointer comparison
        let _ = PartialEq::ne(&a, &b);
        //~^ WARN ambiguous wide pointer comparison
        let _ = a.eq(b);
        //~^ WARN ambiguous wide pointer comparison
        let _ = a.ne(b);
        //~^ WARN ambiguous wide pointer comparison
    }

    let s = "" as *const str;
    let _ = s == s;
    //~^ WARN ambiguous wide pointer comparison

    let s = &[8, 7][..] as *const [i32];
    let _ = s == s;
    //~^ WARN ambiguous wide pointer comparison

    fn cmp<T: ?Sized>(a: *mut T, b: *mut T) -> bool {
        let _ = a == b;
        //~^ WARN ambiguous wide pointer comparison
        let _ = a != b;
        //~^ WARN ambiguous wide pointer comparison
        let _ = a < b;
        //~^ WARN ambiguous wide pointer comparison
        let _ = a <= b;
        //~^ WARN ambiguous wide pointer comparison
        let _ = a > b;
        //~^ WARN ambiguous wide pointer comparison
        let _ = a >= b;
        //~^ WARN ambiguous wide pointer comparison

        let _ = PartialEq::eq(&a, &b);
        //~^ WARN ambiguous wide pointer comparison
        let _ = PartialEq::ne(&a, &b);
        //~^ WARN ambiguous wide pointer comparison
        let _ = a.eq(&b);
        //~^ WARN ambiguous wide pointer comparison
        let _ = a.ne(&b);
        //~^ WARN ambiguous wide pointer comparison

        let a = &a;
        let b = &b;
        &*a == &*b
        //~^ WARN ambiguous wide pointer comparison
    }

    {
        macro_rules! cmp {
            ($a:tt, $b:tt) => { $a == $b }
        }

        // FIXME: This lint uses some custom span combination logic.
        // Rewrite it to adapt to the new metavariable span rules.
        cmp!(a, b);
        //~^ WARN ambiguous wide pointer comparison
    }

    {
        macro_rules! cmp {
            ($a:ident, $b:ident) => { $a == $b }
            //~^ WARN ambiguous wide pointer comparison
        }

        cmp!(a, b);
    }

    {
        // this produce weird diagnostics
        macro_rules! cmp {
            ($a:expr, $b:expr) => { $a == $b }
            //~^ WARN ambiguous wide pointer comparison
        }

        cmp!(&a, &b);
    }

    let _ = std::ptr::eq(a, b);
    let _ = std::ptr::addr_eq(a, b);
    let _ = a as *const () == b as *const ();

    let a: Rc<dyn std::fmt::Debug> = Rc::new(1);
    Rc::ptr_eq(&a, &a);

    let a: Arc<dyn std::fmt::Debug> = Arc::new(1);
    Arc::ptr_eq(&a, &a);
}
