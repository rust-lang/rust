// revisions: classic next
//[next] compile-flags: -Ztrait-solver=next
//check-pass

fn ice()
where
    for<'w> fn(&'w ()): Fn(&'w ()),
{
}

fn main() {
    ice();
}
