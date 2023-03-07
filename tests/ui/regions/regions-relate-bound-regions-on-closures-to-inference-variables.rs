// run-pass
#![allow(dead_code)]
// Test that this fairly specialized, but also reasonable, pattern
// typechecks. The pattern involves regions bound in closures that
// wind up related to inference variables.
//
// NB. Changes to the region implementations have broken this pattern
// a few times, but it happens to be used in the compiler so those
// changes were caught. However, those uses in the compiler could
// easily get changed or refactored away in the future.

struct Ctxt<'tcx> {
    x: &'tcx Vec<isize>
}

struct Foo<'a,'tcx:'a> {
    cx: &'a Ctxt<'tcx>,
}

impl<'a,'tcx> Foo<'a,'tcx> {
    fn bother(&mut self) -> isize {
        self.elaborate_bounds(Box::new(|this| {
            // (*) Here: type of `this` is `&'f0 Foo<&'f1, '_2>`,
            // where `'f0` and `'f1` are fresh, free regions that
            // result from the bound regions on the closure, and `'2`
            // is a region inference variable created by the call. Due
            // to the constraints on the type, we find that `'_2 : 'f1
            // + 'f2` must hold (and can be assumed by the callee).
            // Region inference has to do some clever stuff to avoid
            // inferring `'_2` to be `'static` in this case, because
            // it is created outside the closure but then related to
            // regions bound by the closure itself. See the
            // `region_constraints.rs` file (and the `givens` field, in
            // particular) for more details.
            this.foo()
        }))
    }

    fn foo(&mut self) -> isize {
        22
    }

    fn elaborate_bounds(
        &mut self,
        mut mk_cand: Box<dyn for<'b> FnMut(&mut Foo<'b, 'tcx>) -> isize>)
        -> isize
    {
        mk_cand(self)
    }
}

fn main() {
    let v = vec![];
    let cx = Ctxt { x: &v };
    let mut foo = Foo { cx: &cx };
    assert_eq!(foo.bother(), 22); // just so the code is not dead, basically
}
