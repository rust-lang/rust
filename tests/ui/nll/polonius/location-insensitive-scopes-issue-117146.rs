// This is a non-regression test for issue #117146, where NLL and `-Zpolonius=next` computed
// different loan scopes when a region flowed into an SCC whose representative was an existential
// region.

//@ revisions: nll polonius
//@ [polonius] compile-flags: -Zpolonius=next

fn main() {
    let a = ();
    let b = |_| &a;
    //[nll]~^ ERROR `a` does not live long enough
    //[polonius]~^^ ERROR `a` does not live long enough
    bad(&b);
    //[nll]~^ ERROR implementation of `Fn`
    //[nll]~| ERROR implementation of `FnOnce`
    //[polonius]~^^^ ERROR implementation of `Fn`
    //[polonius]~| ERROR implementation of `FnOnce`
}

fn bad<F: Fn(&()) -> &()>(_: F) {}
