// An example (from @steffahn) of reachability as an approximation of liveness where the polonius
// alpha analysis shows the same imprecision as NLLs, unlike the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

use std::cell::Cell;

struct Invariant<'l>(Cell<&'l ()>);

fn create_invariant<'l>() -> Invariant<'l> {
    Invariant(Cell::new(&()))
}

fn use_it<'a, 'b>(choice: bool) -> Result<Invariant<'a>, Invariant<'b>> {
    let returned_value = create_invariant();
    if choice { Ok(returned_value) } else { Err(returned_value) }
    //[nll]~^ ERROR lifetime may not live long enough
    //[nll]~| ERROR lifetime may not live long enough
    //[polonius]~^^^ ERROR lifetime may not live long enough
    //[polonius]~| ERROR lifetime may not live long enough
}

fn use_it_but_its_the_same_region<'a: 'b, 'b: 'a>(
    choice: bool,
) -> Result<Invariant<'a>, Invariant<'b>> {
    let returned_value = create_invariant();
    if choice { Ok(returned_value) } else { Err(returned_value) }
}

fn main() {}
