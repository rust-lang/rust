// revisions: classic next
//[next] compile-flags: -Znext-solver
// check-pass

fn ice()
where
    for<'w> fn(&'w ()): Fn(&'w ()),
{
}

fn main() {
    ice();
}
