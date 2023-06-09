// build-pass (FIXME(62277): could be check-pass?)

#![allow(warnings)]

type Different<'a, 'b> = &'a mut (&'a (), &'b ());
type Same<'a> = Different<'a, 'a>;

fn with_closure_expecting_different<F>(_: F)
    where F: for<'a, 'b> FnOnce(Different<'a, 'b>)
{
}

fn with_closure_expecting_different_anon<F>(_: F)
    where F: FnOnce(Different<'_, '_>)
{
}

fn supplying_nothing_expecting_anon() {
    with_closure_expecting_different_anon(|x: Different| {
    })
}

fn supplying_nothing_expecting_named() {
    with_closure_expecting_different(|x: Different| {
    })
}

fn supplying_underscore_expecting_anon() {
    with_closure_expecting_different_anon(|x: Different<'_, '_>| {
    })
}

fn supplying_underscore_expecting_named() {
    with_closure_expecting_different(|x: Different<'_, '_>| {
    })
}

fn main() { }
