// Regression test for #53548. The `Box<dyn Trait>` type below is
// expanded to `Box<dyn Trait + 'static>`, but the coroutine "witness"
// that results is `for<'r> { Box<dyn Trait + 'r> }`. The WF code was
// encountering an ICE (when debug-assertions were enabled) and an
// unexpected compilation error (without debug-asserions) when trying
// to process this `'r` region bound. In particular, to be WF, the
// region bound must meet the requirements of the trait, and hence we
// got `for<'r> { 'r: 'static }`. This would ICE because the `Binder`
// constructor we were using was asserting that no higher-ranked
// regions were involved (because the WF code is supposed to skip
// those). The error (if debug-asserions were disabled) came because
// we obviously cannot prove that `'r: 'static` for any region `'r`.
// Pursuant with our "lazy WF" strategy for higher-ranked regions, the
// fix is not to require that `for<'r> { 'r: 'static }` holds (this is
// also analogous to what we would do for higher-ranked regions
// appearing within the trait in other positions).
//
//@ check-pass

#![feature(coroutines, stmt_expr_attributes)]

use std::cell::RefCell;
use std::rc::Rc;

trait Trait: 'static {}

struct Store<C> {
    inner: Rc<RefCell<Option<C>>>,
}

fn main() {
    Box::new(#[coroutine] static move || {
        let store = Store::<Box<dyn Trait>> {
            inner: Default::default(),
        };
        yield ();
    });
}
