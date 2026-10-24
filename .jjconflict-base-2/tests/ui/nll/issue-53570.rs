// Regression test for #53570. Here, we need to propagate that `T: 'a`
// but in some versions of NLL we were propagating a stronger
// requirement that `T: 'static`. This arose because we actually had
// to propagate both that `T: 'a` but also `T: 'b` where `'b` is the
// higher-ranked lifetime that appears in the type of the closure
// parameter `x` -- since `'b` cannot be expressed in the caller's
// space, that got promoted th `'static`.
//
//@ check-pass

use std::cell::{RefCell, Ref};

trait AnyVec<'a> {
}

trait GenericVec<T> {
    fn unwrap<'a, 'b>(vec: &'b dyn AnyVec<'a>) -> &'b [T] where T: 'a;
}

struct Scratchpad<'a> {
    buffers: RefCell<Box<dyn AnyVec<'a>>>,
}

impl<'a> Scratchpad<'a> {
    fn get<T: GenericVec<T>>(&self) -> Ref<'_, [T]>
    where T: 'a
    {
        Ref::map(self.buffers.borrow(), |x| T::unwrap(x.as_ref()))
    }
}

fn main() { }
