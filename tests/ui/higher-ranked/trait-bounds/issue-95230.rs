// revisions: old next
//[next] compile-flags: -Ztrait-solver=next
//[old] check-pass
//[next] known-bug: #109764


pub struct Bar
where
    for<'a> &'a mut Self:;

fn main() {}
