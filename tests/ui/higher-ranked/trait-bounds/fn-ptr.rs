//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

fn ice()
where
    for<'w> fn(&'w ()): Fn(&'w ()),
{
}

fn main() {
    ice();
    //[current]~^ ERROR expected a `Fn(&'w ())` closure, found `fn(&'w ())`
}
