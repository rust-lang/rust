//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

fn ice()
where
    for<'w> fn(&'w ()): Fn(&'w ()),
{
}

fn main() {
    ice();
}
