// Polonius requires liveness for some locals that NLL leaves "boring": locals
// containing some region that outlives a universal region but is not universal itself.
//
// This test checks that we correctly compute liveness for NLL-boring locals that
// are live because of a use.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius_next
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius_next] compile-flags: -Zpolonius=next

struct D<'a>(&'a u32);

impl<'a> Drop for D<'a> {
    fn drop(&mut self) {}
}

// The loan of `*x` flows into `slot`'s region, which outlives `'a` and so is boring to NLLs.
fn assigning_into_a_slot<'a>(x: &'a mut u32, slot: &mut Option<D<'a>>) {
    let r: &'a u32 = &*x;
    *slot = Some(D(r));
    *x = 1; //~ ERROR cannot assign to `*x` because it is borrowed
}

// The same, with the borrow confined to an inner scope and the slot cleared afterwards.
fn clearing_the_slot_does_not_release_it<'a>(x: &'a mut u32, slot: &mut Option<D<'a>>) {
    {
        let r: &'a u32 = &*x;
        *slot = Some(D(r));
    }
    *slot = None;
    *x = 1; //~ ERROR cannot assign to `*x` because it is borrowed
}

fn main() {}
