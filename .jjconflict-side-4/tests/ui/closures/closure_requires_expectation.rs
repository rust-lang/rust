//@ check-pass

// `slot` is an out reference.
// We move `state` into a closure, i.e. the closure holds a &'long mut ().
//
// We move it out of the closure again though, in two steps.
// We first assign it to `temp`.
// `temp` has an expected type, so rust inserts a reborrow.
//
// We then move it again, into `slot`, moving it out of the closure.
// The reborrow could've resulted in `temp` having a shorter lifetime.
// Only in borrowck do we require the lifetime in `temp` to also be `'long`.
//
// However, when we analyze upvars, *we don't know that yet*.
// The reborrow gets treated as: a borrow. And the closure gets inferred to be an FnMut.
// This would make the assignment invalid, you cannot move out of an FnMut!
//
// The reason this compiles is that the closure is checked with an `FnOnce` expectation,
// which makes it so despite the upvars not indicating it should be, the closure becomes FnOnce
// and borrowck is happy that we're moving out of the closure.
//
// Note that the `move` keyword only influences how the upvars are moved into the closure.
// It doesn't change whether the closure can be called more than once.
// This example compiles without `move` keyword as long as `state` is moved into the closure in
// some way. This can alternatively be implemented with `let temp = temp` (without expectation),
// since that doesn't insert a reborrow.
fn wrap<'short, 'long>(slot: &'short mut &'long mut (), state: &'long mut ()) {
    fnonce_requirement(move || {
        let temp: &mut _ = /* rust inserts `&mut *` */ state;
        *slot = temp;
    })
}

fn fnonce_requirement<F>(wrap: F)
where
    F: FnOnce(),
{
    todo!()
}

fn main() {}
